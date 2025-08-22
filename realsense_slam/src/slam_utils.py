# Essential utilities extracted from Open3D reconstruction system
import open3d as o3d
import numpy as np
import json
import os
from pathlib import Path


def load_intrinsic_from_matrix(intrinsic_matrix, width, height):
    """Convert intrinsic matrix to Open3D format"""
    if len(intrinsic_matrix.shape) == 2:
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    else:
        fx = fy = intrinsic_matrix[0]  # Assume square pixels if 1D
        cx, cy = width / 2, height / 2

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def create_tensor_intrinsic(intrinsic_matrix, device):
    """Create tensor intrinsic for Open3D tensor operations"""
    return o3d.core.Tensor(intrinsic_matrix, o3d.core.Dtype.Float64).to(device)


def extract_point_cloud(volume, config, file_name=None):
    """Extract point cloud from volume (adapted from Open3D examples)"""
    if hasattr(volume, 'extract_point_cloud'):
        # Tensor engine
        pcd = volume.extract_point_cloud(
            weight_threshold=config.get('surface_weight_thr', 3.0)
        )
        if file_name is not None:
            o3d.io.write_point_cloud(file_name, pcd.to_legacy())
        return pcd.to_legacy()
    else:
        # Legacy engine - extract from triangle mesh
        mesh = volume.extract_triangle_mesh()
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors

        if file_name is not None:
            o3d.io.write_point_cloud(file_name, pcd)
        return pcd


def extract_triangle_mesh(volume, config, file_name=None):
    """Extract triangle mesh from volume (adapted from Open3D examples)"""
    if hasattr(volume, 'extract_triangle_mesh'):
        # Tensor engine
        mesh = volume.extract_triangle_mesh(
            weight_threshold=config.get('surface_weight_thr', 3.0)
        )
        mesh = mesh.to_legacy()
    else:
        # Legacy engine
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

    if file_name is not None:
        o3d.io.write_triangle_mesh(file_name, mesh)

    return mesh


def save_poses(path_trajectory, poses, intrinsic=None):
    """Save poses in Open3D format (adapted from Open3D examples)"""
    if intrinsic is None:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )

    if path_trajectory.endswith('.log'):
        traj = o3d.camera.PinholeCameraTrajectory()
        params = []
        for pose in poses:
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = intrinsic
            param.extrinsic = np.linalg.inv(pose)
            params.append(param)
        traj.parameters = params
        o3d.io.write_pinhole_camera_trajectory(path_trajectory, traj)

    elif path_trajectory.endswith('.json'):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        for pose in poses:
            node = o3d.pipelines.registration.PoseGraphNode()
            node.pose = pose
            pose_graph.nodes.append(node)
        o3d.io.write_pose_graph(path_trajectory, pose_graph)


def get_open3d_config_defaults():
    """Get default Open3D configuration parameters"""
    return {
        'slam': {
            'voxel_size': 0.0058,
            'block_count': 40000,
            'surface_weight_thr': 3.0,
            'trunc_voxel_multiplier': 8.0,
            'depth_scale': 1000.0,
            'depth_min': 0.1,
            'depth_max': 3.0,
            'use_open3d_slam': True,
            'integrate_color': True
        }
    }


def ensure_directories(base_path):
    """Ensure required directories exist"""
    directories = ['sessions', 'meshes', 'trajectories']
    for dir_name in directories:
        (Path(base_path) / dir_name).mkdir(parents=True, exist_ok=True)


class Open3DIntegration:
    """Minimal wrapper for Open3D tensor SLAM integration"""

    def __init__(self, config, device_str='CUDA:0'):
        self.config = config
        self.device = o3d.core.Device(device_str)

        # Open3D SLAM parameters
        slam_config = config.get('slam', {})
        self.voxel_size = slam_config.get('voxel_size', 0.0058)
        self.block_count = slam_config.get('block_count', 40000)
        self.depth_scale = slam_config.get('depth_scale', 1000.0)
        self.depth_max = slam_config.get('depth_max', 3.0)
        self.depth_min = slam_config.get('depth_min', 0.1)
        self.trunc_multiplier = slam_config.get('trunc_voxel_multiplier', 8.0)

        # Initialize components
        self.model = None
        self.input_frame = None
        self.raycast_frame = None

        print(f"Open3D Integration initialized on {device_str}")
        print(f"Voxel size: {self.voxel_size}, Block count: {self.block_count}")

    def initialize_model(self, depth_image, intrinsic):
        """Initialize Open3D SLAM model"""
        T_frame_to_model = o3d.core.Tensor(np.identity(4))

        self.model = o3d.t.pipelines.slam.Model(
            self.voxel_size, 16, self.block_count,
            T_frame_to_model, self.device
        )

        # Initialize frames
        self.input_frame = o3d.t.pipelines.slam.Frame(
            depth_image.rows, depth_image.columns, intrinsic, self.device
        )
        self.raycast_frame = o3d.t.pipelines.slam.Frame(
            depth_image.rows, depth_image.columns, intrinsic, self.device
        )

        print("Open3D SLAM model initialized")

    def process_frame(self, depth, color, current_pose, frame_idx):
        """Process frame with Open3D tensor SLAM"""
        # Convert to Open3D tensors
        depth_tensor = o3d.t.io.read_image_from_numpy(depth).to(self.device)
        color_tensor = o3d.t.io.read_image_from_numpy(color).to(self.device)

        # Set frame data
        self.input_frame.set_data_from_image('depth', depth_tensor)
        self.input_frame.set_data_from_image('color', color_tensor)

        # Track frame to model (skip first frame)
        if frame_idx > 0:
            result = self.model.track_frame_to_model(
                self.input_frame, self.raycast_frame,
                self.depth_scale, self.depth_max, 0.07
            )
            # Update pose based on tracking result
            current_pose = current_pose @ result.transformation.cpu().numpy()

        # Update model
        pose_tensor = o3d.core.Tensor(current_pose)
        self.model.update_frame_pose(frame_idx, pose_tensor)

        # Integrate frame
        self.model.integrate(
            self.input_frame, self.depth_scale, self.depth_max, self.trunc_multiplier
        )

        # Synthesize model frame for next tracking
        self.model.synthesize_model_frame(
            self.raycast_frame, self.depth_scale, self.depth_min,
            self.depth_max, self.trunc_multiplier, False
        )

        return current_pose

    def get_volume(self):
        """Get the voxel grid volume"""
        return self.model.voxel_grid if self.model else None

    def save_volume(self, filename):
        """Save voxel grid to file"""
        if self.model:
            self.model.voxel_grid.save(filename)
            print(f"Volume saved to {filename}")