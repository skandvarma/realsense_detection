#!/usr/bin/env python3
"""
Diagnose RealSense camera issues.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyrealsense2 as rs
from src.utils.logger import get_logger


def diagnose_camera():
    """Diagnose RealSense camera connection issues."""
    logger = get_logger("CameraDiagnosis")

    try:
        # Check for devices
        context = rs.context()
        devices = context.query_devices()

        if len(devices) == 0:
            logger.error("❌ No RealSense devices found")
            return False

        logger.info(f"✓ Found {len(devices)} RealSense device(s)")

        for i, device in enumerate(devices):
            info = {
                'name': device.get_info(rs.camera_info.name),
                'serial': device.get_info(rs.camera_info.serial_number),
                'firmware': device.get_info(rs.camera_info.firmware_version),
                'usb_type': device.get_info(rs.camera_info.usb_type_descriptor)
            }
            logger.info(f"  Device {i}: {info}")

        # Try simple pipeline
        logger.info("Testing basic pipeline...")

        pipeline = rs.pipeline()
        config = rs.config()

        # Enable basic streams with conservative settings
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            profile = pipeline.start(config)
            logger.info("✓ Pipeline started successfully")

            # Try to get a few frames
            for i in range(5):
                logger.info(f"Attempting frame {i + 1}...")

                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                    if frames:
                        color_frame = frames.get_color_frame()
                        depth_frame = frames.get_depth_frame()

                        logger.info(
                            f"  ✓ Frame {i + 1}: Color={color_frame is not None}, Depth={depth_frame is not None}")
                    else:
                        logger.warning(f"  ⚠️ Frame {i + 1}: No frames received")

                except RuntimeError as e:
                    logger.warning(f"  ❌ Frame {i + 1}: {e}")

                time.sleep(0.1)

            pipeline.stop()
            logger.info("✓ Basic pipeline test completed")

            # Test with different configurations
            logger.info("Testing alternative configurations...")

            # Lower resolution
            config2 = rs.config()
            config2.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)  # Lower res, lower FPS

            try:
                pipeline2 = rs.pipeline()
                profile2 = pipeline2.start(config2)

                frames = pipeline2.wait_for_frames(timeout_ms=3000)
                if frames:
                    logger.info("✓ Lower resolution config works")
                else:
                    logger.warning("⚠️ Lower resolution config timeout")

                pipeline2.stop()

            except Exception as e:
                logger.warning(f"⚠️ Lower resolution config failed: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Pipeline start failed: {e}")
            return False

    except Exception as e:
        logger.error(f"❌ Camera diagnosis failed: {e}")
        return False


def suggest_fixes():
    """Suggest potential fixes for camera issues."""
    print("\n" + "=" * 50)
    print("🔧 CAMERA TROUBLESHOOTING SUGGESTIONS:")
    print("=" * 50)
    print("1. 🔌 USB Connection:")
    print("   - Use USB 3.0+ port (blue connector)")
    print("   - Try different USB port")
    print("   - Use shorter, high-quality USB cable")
    print()
    print("2. 🔋 Power:")
    print("   - Connect external power if available")
    print("   - Close other USB-intensive applications")
    print()
    print("3. 📱 Software:")
    print("   - Close Intel RealSense Viewer if open")
    print("   - Update RealSense drivers")
    print("   - Restart the application")
    print()
    print("4. ⚙️ Alternative Solutions:")
    print("   - Test detection with static images (works perfectly)")
    print("   - Use webcam instead of RealSense")
    print("   - Reduce camera resolution/framerate")
    print("=" * 50)


if __name__ == "__main__":
    print("🔍 RealSense Camera Diagnosis")
    print("=" * 30)

    success = diagnose_camera()

    if not success:
        suggest_fixes()
    else:
        print("\n✅ Camera hardware appears functional")
        print("🔧 Issue might be timeout/configuration related")

    sys.exit(0 if success else 1)