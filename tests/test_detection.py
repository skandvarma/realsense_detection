#!/usr/bin/env python3
"""
Example script demonstrating the complete CUDA-optimized detection pipeline.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.utils.logger import get_logger, PerformanceMonitor
from src.camera.realsense_manager import RealSenseManager
from src.camera.depth_processor import DepthProcessor
from src.detection import DetectorFactory, DetectorWrapper, Postprocessor
from src.detection import YOLO_AVAILABLE, DETR_AVAILABLE


def create_test_setup(config_path: str = "config.yaml"):
    """Create simplified test setup."""
    logger = get_logger("DetectionTest")

    try:
        # Load configuration
        config = ConfigManager.load_config(config_path)
        logger.info("Configuration loaded successfully")

        # Initialize camera system (optional)
        camera_manager = None
        depth_processor = None

        try:
            camera_manager = RealSenseManager(config)
            if camera_manager.initialize_camera():
                depth_processor = DepthProcessor(camera_manager, config)
                depth_processor.update_camera_parameters()
                logger.info("Camera system initialized")
            else:
                logger.warning("Camera initialization failed, using test images only")
        except Exception as e:
            logger.warning(f"Camera setup failed: {e}")

<<<<<<< HEAD
        # Initialize detection system
        factory = DetectorFactory()
        detector = factory.create_detector(config)
=======
        # Initialize detection system - YOLO only
        factory = DetectorFactory()
        detector = factory.create_detector(config, 'detr')
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4

        if not detector:
            logger.error("Failed to create YOLO detector")
            return None

        logger.info("YOLO detector created successfully")

        # Initialize postprocessor
        postprocessor = Postprocessor(config, depth_processor)

        return {
            'config': config,
            'camera_manager': camera_manager,
            'depth_processor': depth_processor,
            'detector': detector,  # Single detector instead of wrapper
            'postprocessor': postprocessor,
            'logger': logger
        }

    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        return None


def test_detection_on_image(setup: Dict[str, Any], image_path: str):
    """Test detection on a single image."""
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

        # Perform detection
        start_time = time.time()
        result = detector.detect(image, frame_id=0)
        detection_time = time.time() - start_time

        if not result.success:
            logger.error(f"Detection failed: {result.error_message}")
            return False

        # Apply postprocessing
        enhanced_result = postprocessor.process_detection_result(result, frame_id=0)

        # Log results
        logger.info(f"\n DETECTION COMPLETED SUCCESSFULLY!")
        logger.info(f"   Detection time: {detection_time * 1000:.1f}ms")
        logger.info(f"   Objects found: {len(enhanced_result.detections)}")
        logger.info(f"  ️  Model: {result.model_name}")
        logger.info(f"   Inference FPS: {result.fps:.1f}")
        logger.info(f"   Preprocessing: {result.preprocessing_time * 1000:.1f}ms")
        logger.info(f"   Inference: {result.inference_time * 1000:.1f}ms")
        logger.info(f"   Postprocessing: {result.postprocessing_time * 1000:.1f}ms")

        # Show detailed detection information
        if enhanced_result.detections:
            logger.info(f"\n📋 DETAILED DETECTION RESULTS:")
            logger.info("=" * 60)

            for i, detection in enumerate(enhanced_result.detections):
                x1, y1, x2, y2 = detection.bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height

                logger.info(f" Detection #{i + 1}:")
                logger.info(f"   Class: {detection.class_name}")
                logger.info(f"   Confidence: {detection.confidence:.3f}")
                logger.info(f"   Bbox: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
                logger.info(f"   Size: {width:.0f}x{height:.0f} pixels ({area:.0f} px²)")

                # Show 3D info if available
                if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                    logger.info(
                        f"   3D Center: ({detection.center_3d[0]:.3f}, {detection.center_3d[1]:.3f}, {detection.center_3d[2]:.3f}m)")
                    logger.info(f"  📏 Distance: {detection.distance:.3f}m")
                    logger.info(f"  🎲 Depth Confidence: {detection.depth_confidence:.3f}")

                # Show tracking info if available
                if detection.detection_id is not None:
                    logger.info(f"    Track ID: {detection.detection_id}")

                logger.info("")  # Empty line between detections
        else:
            logger.info("\n No objects detected in this image")

        # Create and save detailed visualization
        vis_image = create_detection_visualization(image, enhanced_result.detections)

        # Save multiple versions
        output_base = Path(image_path).stem

        # Original with detections
        detection_output = f"detection_result_{output_base}.jpg"
        cv2.imwrite(detection_output, vis_image)
        logger.info(f" Detection result saved: {detection_output}")

        # Side-by-side comparison
        comparison_image = create_side_by_side_comparison(image, vis_image)
        comparison_output = f"comparison_{output_base}.jpg"
        cv2.imwrite(comparison_output, comparison_image)
        logger.info(f" Comparison saved: {comparison_output}")

        # Detection info overlay
        info_image = create_info_overlay(vis_image, enhanced_result)
        info_output = f"detailed_{output_base}.jpg"
        cv2.imwrite(info_output, info_image)
        logger.info(f" Detailed info saved: {info_output}")

        # Class summary
        class_counts = {}
        confidence_sums = {}
        for detection in enhanced_result.detections:
            class_name = detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_sums[class_name] = confidence_sums.get(class_name, 0) + detection.confidence

        if class_counts:
            logger.info(" CLASS SUMMARY:")
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


<<<<<<< HEAD
def test_live_detection(setup: Dict[str, Any], duration: float = 10.0, show_video: bool = False,
                        save_frames: bool = False):
=======
def test_live_detection(setup: Dict[str, Any], duration: float = 10.0, show_video: bool = False, save_frames: bool = False):
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4
    """Test live detection with camera feed."""
    logger = setup['logger']
    camera_manager = setup['camera_manager']
    depth_processor = setup['depth_processor']
    detector = setup['detector']
    postprocessor = setup['postprocessor']

    if not camera_manager:
        logger.error("Camera not available for live detection test")
        return False

    try:
        logger.info(f"Starting live detection test for {duration}s")
        if show_video:
<<<<<<< HEAD
            logger.info("Controls:")
            logger.info("  'q': Quit")
            logger.info("  'p': Change text prompt (for Grounding DINO)")
            logger.info("  's': Save frame")

        # Check if using Grounding DINO
        is_grounding_dino = hasattr(detector, 'is_grounding_dino') and detector.is_grounding_dino
        if is_grounding_dino:
            current_prompt = detector.get_current_prompt()
            logger.info(f"🎯 Grounding DINO mode - Current prompt: '{current_prompt}'")

            # Pre-defined prompts for cycling
            predefined_prompts = [
                "person . car . chair . bottle . laptop",
                "dog . cat . bird . animal",
                "phone . cup . book . keys . bag",
                "car . truck . bus . bicycle . motorcycle",
                "person with glasses . person with hat . child",
                "red object . blue object . green object",
                "food . drink . plate . spoon . fork"
            ]
            current_prompt_index = 0

            logger.info("🎮 Hotkeys:")
            logger.info("  'p': Cycle through pre-defined prompts")
            logger.info("  'r': Reset to default prompt")
        else:
            logger.info("🎮 Hotkeys:")

        logger.info("  'q': Quit")
        logger.info("  's': Save frame")
=======
            logger.info("Press 'q' to quit early if running with display")
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4

        start_time = time.time()
        frame_count = 0
        detection_times = []
        total_detections = 0

        # Create output directory for saving frames
        output_dir = Path("detection_output")
        output_dir.mkdir(exist_ok=True)

        while time.time() - start_time < duration:
            # Capture frame
            frames = camera_manager.capture_frames()
            if not frames or not frames.get('frame_valid', False):
                continue

            color_frame = frames.get('color')
            depth_frame = frames.get('depth')

            if color_frame is None:
                continue

            # Convert RGB to BGR for detection
            if color_frame.shape[2] == 3:
                bgr_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = color_frame

            # Perform detection
            detection_start = time.time()
            result = detector.detect(bgr_frame, frame_id=frame_count)

            if result.success:
                # Apply postprocessing with depth if available
                enhanced_result = postprocessor.process_detection_result(
                    result, depth_frame, frame_count
                )

                detection_time = time.time() - detection_start
                detection_times.append(detection_time)
                total_detections += len(enhanced_result.detections)

                # Show detailed detection info every 30 frames
                if frame_count % 30 == 0 or len(enhanced_result.detections) > 0:
                    fps = 1.0 / detection_time if detection_time > 0 else 0
<<<<<<< HEAD

                    # Show current prompt for Grounding DINO
                    prompt_info = ""
                    if is_grounding_dino:
                        prompt_info = f" | Prompt: '{detector.get_current_prompt()}'"

                    logger.info(
                        f"\n🎥 Frame {frame_count}: {len(enhanced_result.detections)} objects detected, {fps:.1f} FPS{prompt_info}")
=======
                    logger.info(
                        f"\n Frame {frame_count}: {len(enhanced_result.detections)} objects detected, {fps:.1f} FPS")
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4

                    # Show each detection with details
                    for i, detection in enumerate(enhanced_result.detections):
                        x1, y1, x2, y2 = detection.bbox
                        det_info = f"   {i + 1}. {detection.class_name}: {detection.confidence:.3f}"
                        det_info += f" at ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})"

                        # Add 3D info if available
                        if hasattr(detection, 'center_3d') and detection.center_3d != (0, 0, 0):
                            det_info += f" |  3D: ({detection.center_3d[0]:.2f}, {detection.center_3d[1]:.2f}, {detection.center_3d[2]:.2f}m)"
                            det_info += f" |  Distance: {detection.distance:.2f}m"
                        # Add tracking ID if available
                        if detection.detection_id is not None:
                            det_info += f" | ID: {detection.detection_id}"

                        logger.info(det_info)

                # Save visualization every 60 frames or when objects detected (if saving enabled)
                if (frame_count % 60 == 0 or len(enhanced_result.detections) > 0) and save_frames:
                    vis_image = create_detection_visualization(bgr_frame, enhanced_result.detections, depth_frame)
                    output_file = str(output_dir / f"detection_frame_{frame_count:06d}.jpg")
                    cv2.imwrite(output_file, vis_image)

                    if frame_count % 60 == 0:
                        logger.info(f"💾 Saved visualization: detection_frame_{frame_count:06d}.jpg")

                # Show video if requested
                if show_video:
                    vis_image = create_detection_visualization(bgr_frame, enhanced_result.detections, depth_frame)
<<<<<<< HEAD

                    # Add prompt info to visualization for Grounding DINO
                    if is_grounding_dino:
                        prompt_text = f"Prompt: {detector.get_current_prompt()}"
                        cv2.putText(vis_image, prompt_text, (10, vis_image.shape[0] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

=======
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4
                    cv2.imshow("Live Detection", vis_image)

            frame_count += 1

<<<<<<< HEAD
            # Check for early exit and hotkeys
=======
            # Check for early exit (if someone presses 'q' in video window)
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4
            if show_video:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Early exit requested")
                    break
<<<<<<< HEAD
                elif key == ord('p') and is_grounding_dino:
                    # Cycle through pre-defined prompts
                    current_prompt_index = (current_prompt_index + 1) % len(predefined_prompts)
                    new_prompt = predefined_prompts[current_prompt_index]
                    detector.update_text_prompt(new_prompt)
                    logger.info(
                        f"🔄 Switched to prompt {current_prompt_index + 1}/{len(predefined_prompts)}: '{new_prompt}'")
                elif key == ord('r') and is_grounding_dino:
                    # Reset to default prompt
                    detector.update_text_prompt(detector.default_prompt)
                    current_prompt_index = 0
                    logger.info(f"🔄 Reset to default prompt: '{detector.default_prompt}'")
                elif key == ord('s'):
                    # Save current frame
                    if 'enhanced_result' in locals():
                        vis_image = create_detection_visualization(bgr_frame, enhanced_result.detections, depth_frame)
                        save_path = str(output_dir / f"manual_save_{frame_count:06d}.jpg")
                        cv2.imwrite(save_path, vis_image)
                        logger.info(f"💾 Manual save: {save_path}")
=======
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4

        # Final statistics
        total_time = time.time() - start_time
        if frame_count > 0:
            avg_fps = frame_count / total_time
            avg_detection_time = np.mean(detection_times) if detection_times else 0
            detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        else:
            avg_fps = 0
            avg_detection_time = 0
            detection_fps = 0

        logger.info(f"\n LIVE DETECTION TEST COMPLETED!")
        logger.info(f"   Total frames processed: {frame_count}")
        logger.info(f"   Total objects detected: {total_detections}")
        logger.info(f"   Average camera FPS: {avg_fps:.2f}")
        logger.info(f"   Average detection FPS: {detection_fps:.2f}")
<<<<<<< HEAD
        logger.info(f"    Average detection time: {avg_detection_time * 1000:.1f}ms")
=======
        logger.info(f"  ️  Average detection time: {avg_detection_time * 1000:.1f}ms")
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4
        logger.info(f"   Visualizations saved to: {output_dir}/")

        # Get tracking statistics
        tracking_stats = postprocessor.get_tracking_statistics()
        logger.info(f"    Active tracks: {tracking_stats['confirmed_tracks']}")
<<<<<<< HEAD
        logger.info(f"   Total tracks created: {tracking_stats['total_tracks']}")
=======
        logger.info(f"  Total tracks created: {tracking_stats['total_tracks']}")
>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4

        if show_video:
            cv2.destroyAllWindows()

        return True

    except Exception as e:
        logger.error(f"Live detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


<<<<<<< HEAD
=======

>>>>>>> 5b07fbd0f216d193e26203206ab0f90b7f4460b4
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


def benchmark_detectors(setup: Dict[str, Any], test_images: List[np.ndarray]):
    """Benchmark all available detectors."""
    logger = setup['logger']
    config = setup['config']

    try:
        factory = DetectorFactory()
        results = factory.benchmark_all_detectors(config, test_images)

        if not results:
            logger.error("No benchmark results available")
            return False

        logger.info("=== Detector Benchmark Results ===")

        for model_type, result in results.items():
            logger.info(f"{model_type.upper()}:")
            logger.info(f"  Average FPS: {result['avg_fps']:.2f}")
            logger.info(f"  Inference time: {result['avg_inference_time'] * 1000:.2f}ms")
            logger.info(f"  Memory usage: {result['avg_memory_usage_mb']:.1f}MB")
            logger.info(f"  Detections per frame: {result['avg_detections_per_frame']:.1f}")

        # Get recommendation
        recommended = factory.get_recommended_detector({'min_fps': 20})
        if recommended:
            logger.info(f"Recommended detector: {recommended}")

        return True

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return False


def visualize_detections(image: np.ndarray, detections: List, output_path: str):
    """Enhanced visualization of detections on image and save."""
    vis_image = create_detection_visualization(image, detections)
    cv2.imwrite(output_path, vis_image)
    print(f" Enhanced visualization saved to: {output_path}")


def create_test_images() -> List[np.ndarray]:
    """Create synthetic test images for benchmarking."""
    test_images = []

    for i in range(10):
        # Create random test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Add some simple geometric shapes as "objects"
        cv2.rectangle(image, (50 + i * 20, 50), (150 + i * 20, 150), (255, 0, 0), -1)
        cv2.circle(image, (300, 200 + i * 10), 30, (0, 255, 0), -1)

        test_images.append(image)

    return test_images


def main():
    """Main function for detection testing."""
    parser = argparse.ArgumentParser(description="CUDA-Optimized Detection System Test")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--image", "-i",
                        help="Test detection on specific image")
    parser.add_argument("--live", "-l", action="store_true",
                        help="Run live detection test")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Benchmark all available detectors")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                        help="Duration for live test (seconds)")
    parser.add_argument("--show-video", action="store_true", help="Display live video during detection")
    parser.add_argument("--save-frames", action="store_true", help="Save detection frames during live detection")

    args = parser.parse_args()

    # Setup detection system
    setup = create_test_setup(args.config)
    if not setup:
        print("Failed to create test setup")
        sys.exit(1)

    logger = setup['logger']
    success = True

    try:
        if args.image:
            # Test on specific image
            if not os.path.exists(args.image):
                logger.error(f"Image file not found: {args.image}")
                success = False
            else:
                success = test_detection_on_image(setup, args.image)

        elif args.live:
            # Live detection test
            success = test_live_detection(setup, args.duration, show_video=args.show_video, save_frames=args.save_frames)

        elif args.benchmark:
            # Benchmark detectors
            test_images = create_test_images()
            success = benchmark_detectors(setup, test_images)

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

            success = test_detection_on_image(setup, "comprehensive_test_image.jpg")

            if success:
                logger.info("\n COMPREHENSIVE TEST COMPLETED!")
                logger.info(" Check these output files:")
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