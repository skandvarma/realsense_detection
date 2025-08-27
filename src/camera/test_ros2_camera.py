#!/usr/bin/env python3
"""
Test script to verify ROS2 camera integration is working before running full detection system.
"""

import sys
import time
import cv2
import numpy as np

# Add the modified CameraShareManager to path
# You'll need to save the modified CameraShareManager code to a file called 'ros2_camera_manager.py'
from realsense_manager import CameraShareManager, get_logger


def test_ros2_camera():
    """Test ROS2 camera integration."""
    logger = get_logger("ROS2CameraTest")

    logger.info("=== ROS2 Camera Integration Test ===")

    # Create minimal config
    config = {
        'ros2': {
            'color_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info'
        },
        'camera': {
            'width': 640,
            'height': 480,
            'fps': 30
        }
    }

    # Initialize camera manager
    logger.info("Initializing CameraShareManager...")
    camera_manager = CameraShareManager()

    if not camera_manager.initialize_camera(config):
        logger.error("Failed to initialize camera manager!")
        return False

    # Register as test subscriber
    subscriber_id = camera_manager.register_subscriber("TestScript")
    logger.info(f"Registered as subscriber: {subscriber_id}")

    # Wait for camera info
    logger.info("Waiting for camera intrinsics...")
    timeout = 10
    start_time = time.time()
    while camera_manager.intrinsics is None and time.time() - start_time < timeout:
        time.sleep(0.1)

    if camera_manager.intrinsics:
        logger.info(
            f"Camera intrinsics received: fx={camera_manager.intrinsics.fx:.2f}, fy={camera_manager.intrinsics.fy:.2f}")
    else:
        logger.warning("Timeout waiting for camera intrinsics")

    # Test frame capture
    logger.info("Testing frame capture...")
    frame_count = 0
    test_duration = 10  # seconds
    start_time = time.time()

    cv2.namedWindow("ROS2 Camera Test", cv2.WINDOW_NORMAL)

    while time.time() - start_time < test_duration:
        # Get frames
        color, depth = camera_manager.get_frames_for_subscriber(subscriber_id)

        if color is not None and depth is not None:
            frame_count += 1

            # Display info every second
            if frame_count % 30 == 1:
                logger.info(f"Frames received: {frame_count}")
                logger.info(f"Color shape: {color.shape}, Depth shape: {depth.shape}")
                logger.info(f"Status: {camera_manager.get_device_status()}")

            # Create visualization
            # Normalize depth for display
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Stack color and depth side by side
            display = np.hstack((color, depth_colormap))

            # Add text overlay
            cv2.putText(display, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"FPS: {frame_count / (time.time() - start_time):.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("ROS2 Camera Test", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
        else:
            logger.warning("No frames received")
            time.sleep(0.1)

    cv2.destroyAllWindows()

    # Summary
    duration = time.time() - start_time
    fps = frame_count / duration if duration > 0 else 0

    logger.info("=== Test Summary ===")
    logger.info(f"Test duration: {duration:.1f} seconds")
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Average FPS: {fps:.1f}")
    logger.info(f"Final status: {camera_manager.get_device_status()}")

    # Cleanup
    camera_manager.unregister_subscriber(subscriber_id)

    return frame_count > 0


def main():
    """Main test function."""
    # First check if ROS2 topics are available
    import subprocess
    logger = get_logger("ROS2CameraTest")

    logger.info("Checking for ROS2 camera topics...")
    try:
        result = subprocess.run(['ros2', 'topic', 'list'],
                                capture_output=True, text=True, timeout=5)
        topics = result.stdout

        if '/camera/camera/color/image_raw' not in topics:
            logger.error("Camera topics not found! Make sure the RealSense ROS2 node is running.")
            logger.info("Run: ros2 launch realsense2_camera rs_launch.py")
            return 1
        else:
            logger.info("Camera topics found!")

    except Exception as e:
        logger.error(f"Failed to check ROS2 topics: {e}")
        return 1

    # Run the test
    success = test_ros2_camera()

    if success:
        logger.info("ROS2 camera integration test PASSED!")
        return 0
    else:
        logger.error("ROS2 camera integration test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())