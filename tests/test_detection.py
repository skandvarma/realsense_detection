#!/usr/bin/env python3
"""
Example script demonstrating the complete CUDA-optimized detection pipeline with Grounding DINO support.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.logger import get_logger
from src.camera.realsense_manager import CameraShareManager, RealSenseDetectionCamera  # Use ROS2 camera manager
from src.camera.depth_processor import DepthProcessor
from src.detection import DetectorFactory, Postprocessor

import whisper
import sounddevice as sd
import tempfile
import wave

object_detected = False

def set_detection_flag(detected):
    """Write detection status to file."""
    try:
        with open("detection_flag.txt", "w") as f:
            f.write("1" if detected else "0")
    except:
        pass

def get_prompt_from_speech(duration: int = 5, samplerate: int = 16000, model_name: str = "medium") -> str:
    """Capture microphone audio and transcribe it locally with Whisper (English only)."""
    print(f"Recording {duration} seconds of audio for prompt...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    # Save temporary WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav_path = f.name
        with wave.open(f, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())

    # Load Whisper model (choose: "tiny", "base", "small", "medium", "large")
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    # Transcribe in English only
    result = model.transcribe(wav_path, language="en")
    os.remove(wav_path)

    prompt_text = result["text"].strip()
    print(f"Whisper prompt (EN): {prompt_text}")
    return prompt_text


def create_test_setup(config_path: str = "config.yaml"):
    """Create ROS2-compatible test setup."""
    logger = get_logger("DetectionTest")

    try:
        # Load configuration
        config = ConfigManager.load_config(config_path)
        logger.info("Configuration loaded successfully")

        # Initialize ROS2 camera system
        camera_manager = None
        depth_processor = None
        subscriber_id = None

        try:
            logger.info("Initializing ROS2 camera system...")

            # Use CameraShareManager (ROS2-based) instead of RealSenseManager (hardware-based)
            camera_manager = CameraShareManager()

            # Initialize with ROS2 topics
            if camera_manager.initialize_camera(config):
                # Register as subscriber to get frames
                subscriber_id = camera_manager.register_subscriber("DetectionTest")
                logger.info(f"Registered as ROS2 subscriber: {subscriber_id}")

                # Initialize depth processor
                depth_processor = DepthProcessor(camera_manager, config)
                depth_processor.update_camera_parameters()
                logger.info("ROS2 camera system and depth processor initialized")
            else:
                logger.warning("ROS2 camera initialization failed")
                camera_manager = None

        except Exception as e:
            logger.warning(f"ROS2 camera setup failed: {e}")
            logger.info("Make sure ROS2 RealSense node is running:")
            logger.info("  ros2 launch realsense2_camera rs_launch.py")
            camera_manager = None

        # Initialize detection system (unchanged)
        factory = DetectorFactory()
        detector = factory.create_detector(config)

        if not detector:
            logger.error("Failed to create detector")
            return None

        logger.info("Detector created successfully")

        # Initialize postprocessor
        postprocessor = Postprocessor(config, depth_processor)

        return {
            'config': config,
            'camera_manager': camera_manager,
            'depth_processor': depth_processor,
            'detector': detector,
            'postprocessor': postprocessor,
            'logger': logger,
            'subscriber_id': subscriber_id  # Store subscriber ID for cleanup
        }

    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_detection_on_image(setup: Dict[str, Any], image_path: str, text_prompt: str = None):
    """Test detection on a single image."""
    global object_detected
    logger = setup['logger']
    detector = setup['detector']
    postprocessor = setup['postprocessor']

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False

        logger.info(f"Testing detection on image: {image_path}")
        logger.info(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

        # Check if this is a Grounding DINO detector
        is_grounding_dino = hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino
        if is_grounding_dino:
            if not text_prompt:
                text_prompt = get_prompt_from_speech()
                if not text_prompt:
                    logger.error("No prompt provided for Grounding DINO")
                    return False
            detector.update_text_prompt(text_prompt)
            current_prompt = detector.get_current_prompt()
            logger.info(f"Using Grounding DINO with prompt: '{current_prompt}'")

        # Perform detection
        start_time = time.time()
        kwargs = {'frame_id': 0}
        if is_grounding_dino:
            # Always pass text_prompt for Grounding DINO
            kwargs['text_prompt'] = text_prompt or detector.get_current_prompt()

        result = detector.detect(image, **kwargs)
        detection_time = time.time() - start_time

        if not result.success:
            logger.error(f"Detection failed: {result.error_message}")
            return False

        # Apply postprocessing
        enhanced_result = postprocessor.process_detection_result(result, frame_id=0)

        # Set global detection flag
        object_detected = len(enhanced_result.detections) > 0

        # Log results
        logger.info(f"\nDETECTION COMPLETED SUCCESSFULLY!")
        logger.info(f"Detection time: {detection_time * 1000:.1f}ms")
        logger.info(f"Objects found: {len(enhanced_result.detections)}")
        logger.info(f"Model: {result.model_name}")
        logger.info(f"Inference FPS: {result.fps:.1f}")
        logger.info(f"Preprocessing: {result.preprocessing_time * 1000:.1f}ms")
        logger.info(f"Inference: {result.inference_time * 1000:.1f}ms")
        logger.info(f"Postprocessing: {result.postprocessing_time * 1000:.1f}ms")

        if is_grounding_dino:
            logger.info(f"Text prompt used: '{result.metadata.get('text_prompt', 'N/A')}'")

        # Show detailed detection information
        if enhanced_result.detections:
            logger.info(f"\nDETAILED DETECTION RESULTS:")
            logger.info("=" * 60)

            for i, detection in enumerate(enhanced_result.detections):
                x1, y1, x2, y2 = detection.bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height

                logger.info(f"Detection #{i + 1}:")
                logger.info(f"   Class: {detection.class_name}")
                logger.info(f"   Confidence: {detection.confidence:.3f}")
                logger.info(f"   Bbox: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
                logger.info(f"   Size: {width:.0f}x{height:.0f} pixels ({area:.0f} px²)")

                # Show 3D info if available
                if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                    logger.info(
                        f"   3D Center: ({detection.center_3d[0]:.3f}, {detection.center_3d[1]:.3f}, {detection.center_3d[2]:.3f}m)")
                    logger.info(f"   Distance: {detection.distance:.3f}m")
                    logger.info(f"   Depth Confidence: {detection.depth_confidence:.3f}")

                # Show tracking info if available
                if detection.detection_id is not None:
                    logger.info(f"   Track ID: {detection.detection_id}")

                logger.info("")  # Empty line between detections
        else:
            logger.info("\nNo objects detected in this image")

        # Create and save detailed visualization
        vis_image = create_detection_visualization(image, enhanced_result.detections)

        # Save multiple versions
        output_base = Path(image_path).stem

        # Original with detections
        detection_output = f"detection_result_{output_base}.jpg"
        cv2.imwrite(detection_output, vis_image)
        logger.info(f"Detection result saved: {detection_output}")

        # Side-by-side comparison
        comparison_image = create_side_by_side_comparison(image, vis_image)
        comparison_output = f"comparison_{output_base}.jpg"
        cv2.imwrite(comparison_output, comparison_image)
        logger.info(f"Comparison saved: {comparison_output}")

        # Detection info overlay
        info_image = create_info_overlay(vis_image, enhanced_result)
        info_output = f"detailed_{output_base}.jpg"
        cv2.imwrite(info_output, info_image)
        logger.info(f"Detailed info saved: {info_output}")

        # Class summary
        class_counts = {}
        confidence_sums = {}
        for detection in enhanced_result.detections:
            class_name = detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_sums[class_name] = confidence_sums.get(class_name, 0) + detection.confidence

        if class_counts:
            logger.info("CLASS SUMMARY:")
            logger.info("-" * 40)
            for class_name, count in sorted(class_counts.items()):
                avg_confidence = confidence_sums[class_name] / count
                logger.info(f"  {class_name}: {count}x (avg confidence: {avg_confidence:.3f})")

        return True

    except Exception as e:
        logger.error(f"Image detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_live_detection(setup: Dict[str, Any], duration: float = 10.0, show_video: bool = False,
                        save_frames: bool = False):
    """Test live detection with ROS2 camera feed."""
    logger = setup['logger']
    camera_manager = setup['camera_manager']
    depth_processor = setup['depth_processor']
    detector = setup['detector']
    postprocessor = setup['postprocessor']
    subscriber_id = setup.get('subscriber_id')

    if not camera_manager or not subscriber_id:
        logger.error("ROS2 camera not available for live detection test")
        logger.info("Make sure to run: ros2 launch realsense2_camera rs_launch.py")
        return False

    try:
        logger.info(f"Starting ROS2 live detection test for {duration}s")
        logger.info(f"Subscriber ID: {subscriber_id}")

        # Check if using Grounding DINO
        is_grounding_dino = hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino
        if is_grounding_dino:
            user_prompt = get_prompt_from_speech()
            if not user_prompt:
                logger.error("No prompt provided for Grounding DINO")
                return False
            detector.update_text_prompt(user_prompt)
            logger.info(f"Grounding DINO mode - Using prompt: '{detector.get_current_prompt()}'")

        if show_video:
            logger.info("Controls:")
            logger.info("  'q': Quit")
            logger.info("  's': Save frame")
            if is_grounding_dino:
                logger.info("  'p': Cycle through prompts")

        start_time = time.time()
        frame_count = 0
        detection_times = []
        total_detections = 0

        # Create output directory
        output_dir = Path("detection_output")
        output_dir.mkdir(exist_ok=True)

        # Wait for camera to start streaming
        logger.info("Waiting for ROS2 camera frames...")
        wait_start = time.time()
        while time.time() - wait_start < 10.0:  # 10 second timeout
            color_frame, depth_frame = camera_manager.get_frames_for_subscriber(subscriber_id)
            if color_frame is not None:
                logger.info("ROS2 frames received, starting detection...")
                break
            time.sleep(0.1)
        else:
            logger.error("No frames received from ROS2 camera within 10 seconds")
            logger.info("Check that ROS2 RealSense node is publishing to topics:")
            logger.info("  ros2 topic list | grep camera")
            return False

        while time.time() - start_time < duration:
            # Get frames from ROS2 subscriber (KEY CHANGE: different data structure)
            color_frame, depth_frame = camera_manager.get_frames_for_subscriber(subscriber_id)

            if color_frame is None:
                time.sleep(0.01)  # Small delay if no frame
                continue

            # Frame data type conversion (if needed)
            # ROS2 frames might be in different format than direct hardware frames
            if color_frame.shape[2] == 3:
                # Convert RGB to BGR for detection (OpenCV format)
                bgr_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = color_frame

            # Perform detection (unchanged interface)
            detection_start = time.time()
            kwargs = {'frame_id': frame_count}
            if is_grounding_dino:
                kwargs['text_prompt'] = detector.get_current_prompt()

            result = detector.detect(bgr_frame, **kwargs)

            if result.success:
                # Apply postprocessing with depth (depth_frame is numpy array, not dict)
                enhanced_result = postprocessor.process_detection_result(
                    result, depth_frame, frame_count
                )

                global object_detected
                # Set detection flag
                detected = len(enhanced_result.detections) > 0
                set_detection_flag(detected)

                detection_time = time.time() - detection_start
                detection_times.append(detection_time)
                total_detections += len(enhanced_result.detections)

                # Log detection info
                if frame_count % 30 == 0 or len(enhanced_result.detections) > 0:
                    fps = 1.0 / detection_time if detection_time > 0 else 0
                    logger.info(f"Frame {frame_count}: {len(enhanced_result.detections)} objects, {fps:.1f} FPS")

                    for i, detection in enumerate(enhanced_result.detections):
                        x1, y1, x2, y2 = detection.bbox
                        det_info = f"   {i + 1}. {detection.class_name}: {detection.confidence:.3f}"

                        # Add 3D info if available
                        if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                            det_info += f" | 3D: ({detection.center_3d[0]:.2f}, {detection.center_3d[1]:.2f}, {detection.center_3d[2]:.2f}m)"

                        logger.info(det_info)

                # Save frames if requested
                if save_frames and (frame_count % 60 == 0 or len(enhanced_result.detections) > 0):
                    vis_image = create_detection_visualization(bgr_frame, enhanced_result.detections, depth_frame)
                    output_file = str(output_dir / f"ros2_detection_{frame_count:06d}.jpg")
                    cv2.imwrite(output_file, vis_image)

                # Display video if requested
                if show_video:
                    vis_image = create_detection_visualization(bgr_frame, enhanced_result.detections, depth_frame)

                    # Add ROS2 info to display
                    cv2.putText(vis_image, f"ROS2 Frame {frame_count}", (10, vis_image.shape[0] - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if is_grounding_dino:
                        prompt_text = f"Prompt: {detector.get_current_prompt()}"
                        cv2.putText(vis_image, prompt_text, (10, vis_image.shape[0] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv2.imshow("ROS2 Live Detection", vis_image)

            frame_count += 1

            # Handle keyboard input
            if show_video:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Early exit requested")
                    break
                elif key == ord('s'):
                    if 'enhanced_result' in locals():
                        vis_image = create_detection_visualization(bgr_frame, enhanced_result.detections, depth_frame)
                        save_path = str(output_dir / f"ros2_manual_save_{frame_count:06d}.jpg")
                        cv2.imwrite(save_path, vis_image)
                        logger.info(f"Manual save: {save_path}")

        # Final statistics
        total_time = time.time() - start_time
        if frame_count > 0:
            avg_fps = frame_count / total_time
            avg_detection_time = np.mean(detection_times) if detection_times else 0
            detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        else:
            avg_fps = 0
            detection_fps = 0

        logger.info(f"\nROS2 LIVE DETECTION TEST COMPLETED!")
        logger.info(f"Total ROS2 frames processed: {frame_count}")
        logger.info(f"Total objects detected: {total_detections}")
        logger.info(f"Average ROS2 camera FPS: {avg_fps:.2f}")
        logger.info(f"Average detection FPS: {detection_fps:.2f}")

        if show_video:
            cv2.destroyAllWindows()

        return True

    except Exception as e:
        logger.error(f"ROS2 live detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_setup(setup: Dict[str, Any]):
    """Clean up ROS2 test setup resources."""
    logger = setup.get('logger')
    camera_manager = setup.get('camera_manager')
    subscriber_id = setup.get('subscriber_id')
    detector = setup.get('detector')

    try:
        # Cleanup detector
        if detector:
            detector.cleanup()

        # Cleanup ROS2 subscriber
        if camera_manager and subscriber_id:
            camera_manager.unregister_subscriber(subscriber_id)
            logger.info(f"Unregistered ROS2 subscriber: {subscriber_id}")

    except Exception as e:
        if logger:
            logger.warning(f"Cleanup error: {e}")


def create_detection_visualization(image: np.ndarray, detections: List,
                                   depth_frame: Optional[np.ndarray] = None) -> np.ndarray:
    """Create a detailed visualization of detections."""
    vis_image = image.copy()

    # Color palette for different classes
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection.bbox)
        confidence = detection.confidence
        class_name = detection.class_name

        # Choose color based on class or detection index
        color = colors[i % len(colors)]

        # Draw bounding box with thickness based on confidence
        thickness = max(2, int(confidence * 4))
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

        # Create detailed label
        label = f"{class_name}: {confidence:.2f}"
        if hasattr(detection, 'distance') and detection.distance > 0:
            label += f" ({detection.distance:.1f}m)"
        if detection.detection_id is not None:
            label += f" ID:{detection.detection_id}"

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 5, y1), color, -1)

        # Draw label text
        cv2.putText(vis_image, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(vis_image, (center_x, center_y), 3, color, -1)

        # Add 3D position if available
        if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
            pos_text = f"3D: ({detection.center_3d[0]:.1f},{detection.center_3d[1]:.1f},{detection.center_3d[2]:.1f})"
            cv2.putText(vis_image, pos_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add frame info
    frame_info = f"Objects: {len(detections)}"
    cv2.putText(vis_image, frame_info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Add timestamp
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(vis_image, timestamp, (10, vis_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_image


def create_side_by_side_comparison(original: np.ndarray, detection_result: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison of original and detection result."""
    # Ensure both images have same height
    h1, w1 = original.shape[:2]
    h2, w2 = detection_result.shape[:2]

    target_height = min(h1, h2)

    # Resize if needed
    if h1 != target_height:
        original = cv2.resize(original, (int(w1 * target_height / h1), target_height))
    if h2 != target_height:
        detection_result = cv2.resize(detection_result, (int(w2 * target_height / h2), target_height))

    # Add labels
    original_labeled = original.copy()
    detection_labeled = detection_result.copy()

    cv2.putText(original_labeled, "ORIGINAL", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(detection_labeled, "DETECTIONS", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Combine horizontally
    comparison = np.hstack((original_labeled, detection_labeled))

    return comparison


def create_info_overlay(image: np.ndarray, result) -> np.ndarray:
    """Create image with detailed detection information overlay."""
    info_image = image.copy()

    # Add semi-transparent info panel
    overlay = info_image.copy()
    panel_height = min(200, image.shape[0] // 3)
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, info_image, 0.3, 0, info_image)

    # Add text info
    y_offset = 25
    line_height = 25

    cv2.putText(info_image, f"DETECTION RESULTS", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y_offset += line_height + 10

    cv2.putText(info_image, f"Objects Found: {len(result.detections)}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height

    cv2.putText(info_image, f"Model: {result.model_name}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height

    cv2.putText(info_image, f"Inference: {result.inference_time * 1000:.1f}ms", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height

    cv2.putText(info_image, f"FPS: {result.fps:.1f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return info_image


def main():
    """Main function for detection testing."""
    parser = argparse.ArgumentParser(description="CUDA-Optimized Detection System Test with Grounding DINO")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--image", "-i",
                        help="Test detection on specific image")
    parser.add_argument("--prompt", "-p",
                        help="Text prompt for Grounding DINO (e.g., 'person . car . dog')")
    parser.add_argument("--live", "-l", default=True, type=bool,
                        help="Run live detection test (always True)")
    parser.add_argument("--duration", "-d", type=float, default=1000000000.0,
                        help="Duration for live test (seconds)")
    parser.add_argument("--show-video", default=True, type=bool,
                        help="Display live video during detection (always True)")
    parser.add_argument("--save-frames", action="store_true", help="Save detection frames during live detection")

    args = parser.parse_args()

    # Setup detection system
    setup = create_test_setup(args.config)
    if not setup:
        print("Failed to create ROS2 test setup")
        print("Make sure ROS2 RealSense node is running:")
        print("  ros2 launch realsense2_camera rs_launch.py")
        sys.exit(1)

    logger = setup['logger']
    success = True

    try:
        if args.image:
            # Test on specific image (unchanged)
            success = test_detection_on_image(setup, args.image, args.prompt)

        elif args.live:
            # ROS2 live detection test
            logger.info("Starting ROS2 live detection test...")
            logger.info("Make sure ROS2 RealSense node is running:")
            logger.info("  ros2 launch realsense2_camera rs_launch.py")
            success = test_live_detection(setup, args.duration, show_video=args.show_video,
                                          save_frames=args.save_frames)
        else:
            # Default comprehensive test (unchanged)
            success = test_detection_on_image(setup, "comprehensive_test_image.jpg", args.prompt)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        success = False
    finally:
        # ROS2-specific cleanup
        cleanup_test_setup(setup)

    sys.exit(0 if success else 1)

    try:
        if args.image:
            # Test on specific image
            if not os.path.exists(args.image):
                logger.error(f"Image file not found: {args.image}")
                success = False
            else:
                success = test_detection_on_image(setup, args.image, args.prompt)

        elif args.live:
            # Live detection test
            success = test_live_detection(setup, args.duration, show_video=args.show_video,
                                          save_frames=args.save_frames)

        else:
            # Default: create test image and run detection
            logger.info("No specific test specified, running comprehensive detection test")

            # Create a more realistic test image with multiple objects
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Add background
            test_image.fill(100)

            # Add person-like rectangles
            cv2.rectangle(test_image, (100, 150), (180, 400), (150, 100, 80), -1)  # Person 1
            cv2.rectangle(test_image, (400, 180), (480, 420), (120, 90, 70), -1)  # Person 2

            # Add car-like shape
            cv2.rectangle(test_image, (200, 300), (350, 380), (50, 50, 200), -1)  # Car body
            cv2.rectangle(test_image, (210, 280), (340, 300), (80, 80, 250), -1)  # Car roof

            # Add circular objects
            cv2.circle(test_image, (500, 150), 40, (200, 200, 50), -1)  # Ball/object
            cv2.circle(test_image, (550, 350), 25, (50, 200, 200), -1)  # Another object

            # Add some noise/texture
            noise = np.random.randint(-30, 30, test_image.shape, dtype=np.int16)
            test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Save test image
            cv2.imwrite("comprehensive_test_image.jpg", test_image)
            logger.info("Created comprehensive test image with multiple objects")

            success = test_detection_on_image(setup, "comprehensive_test_image.jpg", args.prompt)

            if success:
                logger.info("\nCOMPREHENSIVE TEST COMPLETED!")
                logger.info("Check these output files:")
                logger.info("  - detection_result_comprehensive_test_image.jpg")
                logger.info("  - comparison_comprehensive_test_image.jpg")
                logger.info("  - detailed_comprehensive_test_image.jpg")

        # Performance summary
        if setup['detector']:
            perf_stats = setup['detector'].get_performance_stats()
            logger.info("=== Performance Summary ===")
            for key, value in perf_stats.items():
                logger.info(f"{key}: {value}")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        success = False
    finally:
        # Cleanup
        if setup['detector']:
            setup['detector'].cleanup()
        if setup['camera_manager']:
            setup['camera_manager'].cleanup()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()