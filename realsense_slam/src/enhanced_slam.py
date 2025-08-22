import open3d as o3d
import numpy as np
import json
import cv2
import time
from motion_detector import VisualIMUMotionDetector
from coordinate_aligner import CoordinateFrameAligner


class EnhancedMinimalSLAM:
    def __init__(self, intrinsics, config):
        self.intrinsics = intrinsics
        self.config = config

        # Initialize motion detector and coordinate aligner
        self.motion_detector = VisualIMUMotionDetector()
        self.coordinate_aligner = CoordinateFrameAligner()

        # SLAM parameters
        self.params = {
            'max_depth': config.get('slam', {}).get('max_depth', 3.0),
            'depth_scale': config.get('slam', {}).get('depth_scale', 1000.0),
            'process_every_n': config.get('slam', {}).get('process_every_n', 1),
            'odometry_method': config.get('slam', {}).get('odometry_method', 'hybrid'),
            'voxel_size': config.get('slam', {}).get('voxel_size', 0.01)
        }

        # Visual SLAM state
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        self.frame_count = 0
        self.last_time = time.time()

        # RGB-D frames for odometry
        self.prev_rgbd = None
        self.current_rgbd = None

        # Point cloud map for visualization
        self.map_cloud = o3d.geometry.PointCloud()
        self.trajectory_points = []

        # IMU trajectory tracking
        self.imu_position = np.zeros(3)
        self.imu_velocity = np.zeros(3)
        self.imu_trajectory = [np.zeros(3)]
        self.trajectory_timestamps = [time.time()]

        # Scale alignment
        self.scale_factor = 1.0
        self.scale_estimation_window = 20
        self.last_scale_update = 0
        self.scale_history = []
        self.scale_confidence = 0.0
        self.scale_stability_count = 0
        self.cumulative_visual_distance = 0.0
        self.cumulative_imu_distance = 0.0

        # Motion tracking
        self.motion_log = []
        self.stationary_count = 0

        # Open3D camera intrinsic
        height, width = config['camera']['height'], config['camera']['width']
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2]
        )

        # Open3D odometry option with version compatibility
        self.odometry_option = o3d.pipelines.odometry.OdometryOption()

        # Set available attributes (varies by Open3D version)
        available_attrs = dir(self.odometry_option)

        if 'max_depth_diff' in available_attrs:
            self.odometry_option.max_depth_diff = 0.07
        if 'min_depth' in available_attrs:
            self.odometry_option.min_depth = 0.1
        if 'max_depth' in available_attrs:
            self.odometry_option.max_depth = self.params['max_depth']
        if 'depth_diff_threshold' in available_attrs:
            self.odometry_option.depth_diff_threshold = 0.07
        if 'depth_threshold' in available_attrs:
            self.odometry_option.depth_threshold = [0.1, self.params['max_depth']]

        print(f"Available odometry options: {[attr for attr in available_attrs if not attr.startswith('_')]}")

        # Choose odometry method
        if self.params['odometry_method'] == 'color':
            self.odometry_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()
        elif self.params['odometry_method'] == 'depth':
            self.odometry_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromDepthTerm()
        else:  # hybrid (default)
            self.odometry_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

        print(f"Enhanced SLAM initialized with Open3D {self.params['odometry_method']} odometry")
        print(f"Intrinsics: {self.intrinsic.intrinsic_matrix}")

    def update_imu_trajectory(self, accel, gyro, dt):
        """IMU trajectory tracking with proper coordinate alignment"""
        if accel is None or dt <= 0 or dt > 0.1:
            return

        # Apply coordinate alignment to IMU data
        accel_4d = np.array([accel[0], accel[1], accel[2], 0])
        aligned_accel_4d = self.coordinate_aligner.camera_to_slam @ accel_4d
        aligned_accel = aligned_accel_4d[:3]

        # Remove gravity and integrate
        accel_magnitude = np.linalg.norm(aligned_accel)
        if accel_magnitude > 5.0:
            gravity_removed = max(0, accel_magnitude - 9.8)

            if accel_magnitude > 0:
                accel_direction = aligned_accel / accel_magnitude
                world_accel = accel_direction * gravity_removed
            else:
                world_accel = np.zeros(3)

            self.imu_velocity += world_accel * dt
            self.imu_velocity *= 0.95

            prev_position = self.imu_position.copy()
            self.imu_position += self.imu_velocity * dt

            imu_step_distance = np.linalg.norm(self.imu_position - prev_position)
            self.cumulative_imu_distance += imu_step_distance

            self.imu_trajectory.append(self.imu_position.copy())
            self.trajectory_timestamps.append(time.time())

            if len(self.imu_trajectory) > 1000:
                self.imu_trajectory.pop(0)
                self.trajectory_timestamps.pop(0)

    def estimate_scale_factor(self):
        """Enhanced scale estimation with confidence tracking"""
        if len(self.trajectory) < self.scale_estimation_window:
            return

        visual_poses = self.trajectory[-self.scale_estimation_window:]
        imu_positions = self.imu_trajectory[-self.scale_estimation_window:]

        if len(imu_positions) < self.scale_estimation_window:
            return

        visual_distance = 0.0
        imu_distance = 0.0

        for i in range(1, len(visual_poses)):
            visual_pos1 = visual_poses[i - 1][:3, 3]
            visual_pos2 = visual_poses[i][:3, 3]
            visual_step = np.linalg.norm(visual_pos2 - visual_pos1)
            visual_distance += visual_step
            self.cumulative_visual_distance += visual_step

            imu_pos1 = imu_positions[i - 1]
            imu_pos2 = imu_positions[i]
            imu_distance += np.linalg.norm(imu_pos2 - imu_pos1)

        if visual_distance > 0.01 and imu_distance > 0.01:
            new_scale = imu_distance / visual_distance

            self.scale_history.append(new_scale)
            if len(self.scale_history) > 10:
                self.scale_history.pop(0)

            if len(self.scale_history) >= 3:
                scale_variance = np.var(self.scale_history[-3:])
                self.scale_confidence = max(0.0, 1.0 - scale_variance * 10)

                if scale_variance < 0.01:
                    self.scale_stability_count += 1
                else:
                    self.scale_stability_count = 0

            alpha = 0.1
            self.scale_factor = alpha * new_scale + (1 - alpha) * self.scale_factor

            print(f"Scale factor updated: {self.scale_factor:.3f} "
                  f"(IMU: {imu_distance:.3f}m, Visual: {visual_distance:.3f}m, "
                  f"Confidence: {self.scale_confidence:.2f})")

    def create_rgbd_image(self, rgb, depth):
        """Create Open3D RGB-D image from numpy arrays"""
        # Convert RGB to Open3D format
        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        # Convert depth to Open3D format (ensure correct data type)
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        # Create RGB-D image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,  # Already converted to meters
            depth_trunc=self.params['max_depth'],
            convert_rgb_to_intensity=False
        )

        return rgbd

    def compute_visual_odometry(self, current_rgbd, prev_rgbd):
        """Compute visual odometry using Open3D RGB-D odometry pipeline"""

        # Initial transformation guess
        odo_init = np.eye(4)

        # Compute RGB-D odometry
        success, transformation, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd,  # source (previous frame)
            current_rgbd,  # target (current frame)
            self.intrinsic,  # camera intrinsic
            odo_init,  # initial transformation guess
            self.odometry_method,  # odometry method (color/depth/hybrid)
            self.odometry_option  # odometry options
        )

        return success, transformation, info

    def create_point_cloud_from_rgbd(self, rgbd_image, pose=None):
        """Create point cloud from RGB-D image"""
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsic
        )

        if pose is not None:
            pcd.transform(pose)

        if len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(self.params['voxel_size'])

        return pcd

    def process_frame(self, rgb, depth, accel=None, gyro=None):
        """Enhanced frame processing with Open3D RGB-D odometry"""
        self.frame_count += 1
        current_time = time.time()
        dt = current_time - self.last_time

        # Update IMU trajectory
        self.update_imu_trajectory(accel, gyro, dt)

        # Skip frames for performance if configured
        if self.frame_count % self.params['process_every_n'] != 0:
            self.last_time = current_time
            return

        # Motion detection
        motion_result = self.analyze_motion(rgb, accel, gyro, dt)

        # Log motion analysis
        self.motion_log.append({
            'frame': self.frame_count,
            'timestamp': current_time,
            'motion_detected': motion_result['decision']['has_motion'],
            'visual_motion': motion_result['visual']['has_motion'],
            'imu_motion': motion_result['imu']['has_motion'],
            'agreement': motion_result['decision']['agreement'],
            'should_update': motion_result['decision']['should_update_slam']
        })

        # Process SLAM update based on motion
        if motion_result['decision']['should_update_slam']:
            self.process_slam_update(rgb, depth, motion_result)
        else:
            self.stationary_count += 1
            self.trajectory.append(self.current_pose.copy())

        # Update scale factor periodically
        if self.frame_count - self.last_scale_update > 30:
            self.estimate_scale_factor()
            self.last_scale_update = self.frame_count

        self.last_time = current_time

    def process_slam_update(self, rgb, depth, motion_result):
        """Process SLAM update using Open3D RGB-D odometry"""

        # Create current RGB-D image
        self.current_rgbd = self.create_rgbd_image(rgb, depth)

        if self.prev_rgbd is not None:
            # Compute visual odometry using Open3D
            success, transformation, info = self.compute_visual_odometry(
                self.current_rgbd, self.prev_rgbd
            )

            if success:
                # Validate transformation
                if self.validate_transform(transformation, motion_result):
                    # Update pose
                    self.current_pose = np.dot(self.current_pose, transformation)
                    print(f"Frame {self.frame_count}: Odometry success, "
                          f"translation: {np.linalg.norm(transformation[:3, 3]):.4f}m")
                else:
                    print(f"Frame {self.frame_count}: Transform rejected (too large)")

                self.trajectory.append(self.current_pose.copy())

                # Update trajectory points for visualization
                self.trajectory_points.append(self.current_pose[:3, 3].copy())

            else:
                print(f"Frame {self.frame_count}: Odometry failed")
                self.trajectory.append(self.current_pose.copy())

            # Accumulate point cloud for mapping
            self.accumulate_to_map()

        else:
            # First frame
            self.trajectory.append(self.current_pose.copy())
            self.trajectory_points.append(self.current_pose[:3, 3].copy())
            print(f"Frame {self.frame_count}: First frame initialized")

        # Update for next iteration
        self.prev_rgbd = self.current_rgbd

    def accumulate_to_map(self):
        """Accumulate current frame to point cloud map"""
        if self.current_rgbd is not None:
            # Create point cloud from current RGB-D with current pose
            pcd = self.create_point_cloud_from_rgbd(self.current_rgbd, self.current_pose)

            if len(pcd.points) > 0:
                # Add to map
                self.map_cloud += pcd

                # Downsample map if it gets too large
                if len(self.map_cloud.points) > 100000:
                    self.map_cloud = self.map_cloud.voxel_down_sample(self.params['voxel_size'] * 1.5)
                    print(f"Map downsampled to {len(self.map_cloud.points)} points")

    def analyze_motion(self, rgb, accel, gyro, dt):
        """Analyze motion using visual-IMU detector"""
        visual_motion = self.motion_detector.detect_visual_motion(rgb)
        imu_motion = self.motion_detector.analyze_imu_motion(accel, gyro, dt)
        motion_decision = self.motion_detector.fuse_motion_detection(visual_motion, imu_motion, dt)

        return {
            'visual': visual_motion,
            'imu': imu_motion,
            'decision': motion_decision
        }

    def validate_transform(self, transform, motion_result):
        """Validate transform against motion detection"""
        translation = np.linalg.norm(transform[:3, 3])
        rotation_angle = np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1))

        # Reasonable motion thresholds
        max_translation = 0.3  # 30cm per frame
        max_rotation = 0.5  # ~30 degrees per frame

        return translation <= max_translation and rotation_angle <= max_rotation

    def get_map(self):
        """Get current map point cloud"""
        return self.map_cloud

    def get_trajectory(self):
        """Return scale-aligned visual trajectory"""
        return self.get_aligned_trajectory()

    def get_aligned_trajectory(self):
        """Return visual trajectory aligned with IMU scale"""
        aligned_trajectory = []

        for pose in self.trajectory:
            aligned_pose = pose.copy()
            aligned_pose[:3, 3] *= self.scale_factor
            aligned_trajectory.append(aligned_pose)

        return aligned_trajectory

    def get_imu_trajectory(self):
        """Return IMU trajectory for visualization"""
        return self.get_imu_trajectory_poses()

    def get_imu_trajectory_poses(self):
        """Convert IMU positions to 4x4 pose format for visualization"""
        imu_poses = []

        for position in self.imu_trajectory:
            pose = np.eye(4)
            pose[:3, 3] = position
            imu_poses.append(pose)

        return imu_poses

    def get_scale_info(self):
        """Get comprehensive scale information"""
        scale_quality = "Unknown"
        if self.scale_confidence > 0.8:
            scale_quality = "Excellent"
        elif self.scale_confidence > 0.6:
            scale_quality = "Good"
        elif self.scale_confidence > 0.4:
            scale_quality = "Fair"
        else:
            scale_quality = "Poor"

        return {
            'scale_factor': self.scale_factor,
            'scale_confidence': self.scale_confidence,
            'scale_quality': scale_quality,
            'scale_stability_count': self.scale_stability_count,
            'scale_history': self.scale_history.copy(),
            'cumulative_visual_distance': self.cumulative_visual_distance,
            'cumulative_imu_distance': self.cumulative_imu_distance,
            'total_distance_ratio': self.cumulative_imu_distance / max(0.001, self.cumulative_visual_distance)
        }

    def get_motion_stats(self):
        """Get motion detection statistics with enhanced scale information"""
        if len(self.motion_log) < 10:
            return {}

        recent_log = self.motion_log[-100:]
        total_frames = len(recent_log)
        motion_frames = sum(1 for entry in recent_log if entry['motion_detected'])

        agreements = [entry['agreement'] for entry in recent_log]
        agreement_counts = {}
        for agreement in agreements:
            agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1

        drift_detections = sum(1 for entry in recent_log if entry['agreement'] == 'imu_drift_detected')
        drift_rate = drift_detections / total_frames if total_frames > 0 else 0

        scale_info = self.get_scale_info()

        return {
            'total_frames': total_frames,
            'motion_frames': motion_frames,
            'motion_rate': motion_frames / total_frames if total_frames > 0 else 0,
            'drift_detection_rate': drift_rate,
            'agreement_counts': agreement_counts,
            'scale_factor': self.scale_factor,
            'scale_info': scale_info,
            'slam_mode': f'Open3D {self.params["odometry_method"].upper()}'
        }

    def visualize_trajectory_open3d(self):
        """Visualize trajectory using Open3D visualizer"""
        if len(self.trajectory_points) < 2:
            return

        # Create trajectory line set
        points = o3d.utility.Vector3dVector(self.trajectory_points)
        lines = [[i, i + 1] for i in range(len(self.trajectory_points) - 1)]

        trajectory_lines = o3d.geometry.LineSet()
        trajectory_lines.points = points
        trajectory_lines.lines = o3d.utility.Vector2iVector(lines)

        # Green color for trajectory
        colors = [[0, 1, 0] for _ in range(len(lines))]
        trajectory_lines.colors = o3d.utility.Vector3dVector(colors)

        # Coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Visualize
        geometries = [coord_frame, trajectory_lines]
        if len(self.map_cloud.points) > 0:
            geometries.append(self.map_cloud)

        o3d.visualization.draw_geometries(geometries,
                                          window_name="Open3D SLAM - Trajectory and Map",
                                          width=1024, height=768)

    def save_session(self, filename):
        """Save session with comprehensive data"""
        # Save point cloud map
        o3d.io.write_point_cloud(f"{filename}_map.ply", self.map_cloud)

        # Save trajectory as poses
        trajectory_data = {
            "visual_poses": [pose.tolist() for pose in self.get_aligned_trajectory()],
            "imu_poses": [pose.tolist() for pose in self.get_imu_trajectory()],
            "scale_factor": self.scale_factor,
            "scale_info": self.get_scale_info(),
            "frame_count": self.frame_count,
            "params": self.params,
            "motion_stats": self.get_motion_stats(),
            "motion_log": self.motion_log[-1000:],
            "slam_mode": f'Open3D {self.params["odometry_method"].upper()} Odometry'
        }

        with open(f"{filename}_trajectory.json", 'w') as f:
            json.dump(trajectory_data, f, indent=2)

        with open(f"{filename}_motion_log.json", 'w') as f:
            json.dump(self.motion_log, f, indent=2)

        print(f"Session saved with Open3D {self.params['odometry_method']} odometry:")
        print(f"  Scale factor: {self.scale_factor:.3f}")
        print(f"  Scale confidence: {self.get_scale_info()['scale_confidence']:.2f}")
        print(f"  Map points: {len(self.map_cloud.points)}")
        print(f"  Trajectory length: {len(self.trajectory)} poses")


def main():
    """Test Open3D RGB-D odometry integration"""
    config = {
        "camera": {"width": 640, "height": 480, "fps": 30},
        "slam": {
            "max_depth": 3.0,
            "depth_scale": 1000.0,
            "process_every_n": 1,
            "odometry_method": "hybrid",  # color, depth, or hybrid
            "voxel_size": 0.01
        },
        "viz": {
            "point_size": 2,
            "background": [0, 0, 0]
        }
    }

    from camera import D435iCamera

    print("Testing Open3D RGB-D Odometry Pipeline...")
    camera = D435iCamera(config)
    intrinsics = camera.get_intrinsics()
    slam = EnhancedMinimalSLAM(intrinsics, config)

    print("Waiting for camera to warm up...")
    time.sleep(2)

    frames_processed = 0
    none_frame_count = 0
    max_none_frames = 50

    try:
        for i in range(500):
            rgb, depth = camera.get_frames()
            accel, gyro = camera.get_imu_data()

            if rgb is not None and depth is not None:
                depth_meters = depth.astype(np.float32) / 1000.0
                slam.process_frame(rgb, depth_meters, accel, gyro)
                frames_processed += 1

                if frames_processed % 30 == 0:
                    stats = slam.get_motion_stats()
                    scale_info = stats.get('scale_info', {})
                    print(f"Frame {frames_processed}: "
                          f"Mode: {stats.get('slam_mode', 'Unknown')}, "
                          f"Trajectory: {len(slam.get_trajectory())} poses, "
                          f"Map: {len(slam.get_map().points)} points, "
                          f"Scale: {stats.get('scale_factor', 1.0):.3f}")

                if frames_processed >= 200:
                    break
            else:
                none_frame_count += 1
                if none_frame_count <= 10:
                    print(f"Waiting for frames... (attempt {none_frame_count})")

                if none_frame_count >= max_none_frames:
                    print(f"Camera not providing frames after {max_none_frames} attempts")
                    break

                time.sleep(0.1)

        print(f"Processed {frames_processed} frames total")

        if frames_processed > 0:
            slam.save_session("../data/sessions/open3d_odometry_test")

            # Visualize results
            print("Showing trajectory and map visualization...")
            slam.visualize_trajectory_open3d()

            print("Open3D RGB-D odometry test complete")
        else:
            print("No frames processed")

    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.stop()


if __name__ == "__main__":
    main()