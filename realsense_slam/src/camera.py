# Import D435iCamera from the enhanced realsense_manager
import sys
import os

# Add the project src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Import directly from the file to avoid package issues
realsense_manager_path = os.path.join(src_path, 'camera', 'realsense_manager.py')

if os.path.exists(realsense_manager_path):
    # Load the module directly
    import importlib.util

    spec = importlib.util.spec_from_file_location("realsense_manager", realsense_manager_path)
    realsense_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realsense_manager)

    # Get the D435iCamera class
    D435iCamera = realsense_manager.D435iCamera
    print("Successfully imported D435iCamera from realsense_manager")
else:
    raise ImportError(f"Could not find realsense_manager.py at {realsense_manager_path}")

# Re-export for compatibility
__all__ = ['D435iCamera']


def main():
    """Test function"""
    import json

    # Test with existing config format
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    camera = D435iCamera(config)
    print("Testing enhanced camera... Press 'q' to quit")

    try:
        for i in range(100):
            rgb, depth = camera.get_frames()
            accel, gyro = camera.get_imu_data()

            if rgb is not None and depth is not None:
                print(f"Frame {i}: RGB {rgb.shape}, Depth {depth.shape}")
                if i >= 10:  # Test a few frames then exit
                    break

    finally:
        camera.stop()
        print("Test completed successfully!")


if __name__ == "__main__":
    main()