import numpy as np


class CoordinateFrameAligner:
    def __init__(self):
        """
        RealSense D435i Coordinate Frames:

        Camera Frame (RGB/Depth):
        - X: Right
        - Y: Down
        - Z: Forward (into scene)

        IMU Frame:
        - X: Right
        - Y: Forward (direction camera faces)
        - Z: Up

        Open3D/SLAM Frame (desired):
        - X: Right
        - Y: Forward
        - Z: Up
        """

        # Transformation from Camera frame to IMU frame
        # Camera: X=right, Y=down, Z=forward
        # IMU: X=right, Y=forward, Z=up
        self.camera_to_imu = np.array([
            [1, 0, 0, 0],  # X stays X
            [0, 0, 1, 0],  # Camera Z becomes IMU Y
            [0, -1, 0, 0],  # Camera -Y becomes IMU Z
            [0, 0, 0, 1]
        ])

        # IMU to SLAM coordinate alignment (if needed)
        # This is identity since we want IMU frame as reference
        self.imu_to_slam = np.eye(4)

        # Combined transformation
        self.camera_to_slam = np.dot(self.imu_to_slam, self.camera_to_imu)

        print("Coordinate Frame Aligner initialized")
        print("Camera → IMU → SLAM transformation ready")

    def align_point_cloud(self, pcd):
        """Transform point cloud from camera frame to aligned frame"""
        return pcd.transform(self.camera_to_slam)

    def align_pose(self, pose_camera_frame):
        # MINIMAL FIX: Check if pose represents significant motion
        translation_magnitude = np.linalg.norm(pose_camera_frame[:3, 3])
        if translation_magnitude < 0.005:  # Less than 5mm - likely noise
            return np.eye(4)  # Return identity to prevent noise amplification

        return np.dot(self.camera_to_slam, pose_camera_frame)

    def align_imu_trajectory_to_visual(self, imu_poses):
        """
        Ensure IMU trajectory and visual trajectory are in same coordinate frame
        This is the key function to make them align
        """
        aligned_poses = []
        for pose in imu_poses:
            # IMU pose is already in IMU frame, just ensure it matches SLAM frame
            aligned_pose = np.dot(self.imu_to_slam, pose)
            aligned_poses.append(aligned_pose)
        return aligned_poses

    def create_alignment_visualization(self):
        """Create coordinate frame visualization for debugging"""
        import open3d as o3d

        # Create coordinate frames for visualization
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        camera_frame.paint_uniform_color([1, 0, 0])  # Red for camera

        imu_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        imu_frame.transform(self.camera_to_imu)
        imu_frame.paint_uniform_color([0, 1, 0])  # Green for IMU

        slam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        slam_frame.transform(self.camera_to_slam)
        slam_frame.paint_uniform_color([0, 0, 1])  # Blue for SLAM

        return [camera_frame, imu_frame, slam_frame]


def test_coordinate_alignment():
    """Test coordinate frame alignment"""
    import open3d as o3d

    aligner = CoordinateFrameAligner()

    # Test with sample point cloud
    # Create a simple point cloud in camera frame
    points = np.array([
        [0, 0, 1],  # Point in front of camera
        [1, 0, 1],  # Point to the right
        [0, 1, 1],  # Point below camera
    ])

    pcd_camera = o3d.geometry.PointCloud()
    pcd_camera.points = o3d.utility.Vector3dVector(points)

    # Transform to aligned frame
    pcd_aligned = aligner.align_point_cloud(pcd_camera)

    print("Original points (camera frame):")
    print(np.asarray(pcd_camera.points))
    print("\nAligned points (SLAM frame):")
    print(np.asarray(pcd_aligned.points))

    # Visualize frames
    frames = aligner.create_alignment_visualization()
    o3d.visualization.draw_geometries(frames + [pcd_camera, pcd_aligned])


if __name__ == "__main__":
    test_coordinate_alignment()