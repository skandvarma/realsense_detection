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
            'voxel_size': config.get('slam', {}).get('voxel_size', 0.05),
            'max_points': config.get('slam', {}).get('max_points', 20000),
            'icp_threshold': config.get('slam', {}).get('icp_threshold', 0.02),
            'max_depth': config.get('slam', {}).get('max_depth', 3.0),
            'process_every_n': config.get('slam', {}).get('process_every_n', 1),
            'accumulate_every_n': config.get('slam', {}).get('accumulate_every_n', 3)
        }

        # Visual SLAM state
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        self.map_cloud = o3d.geometry.PointCloud()
        self.prev_cloud = None
        self.prev_rgb = None
        self.frame_count = 0
        self.last_time = time.time()

        # IMU trajectory tracking (MINIMAL ADDITION)
        self.imu_position = np.zeros(3)
        self.imu_velocity = np.zeros(3)
        self.imu_trajectory = [np.zeros(3)]
        self.trajectory_timestamps = [time.time()]

        # Scale alignment (MINIMAL ADDITION)
        self.scale_factor = 1.0
        self.scale_estimation_window = 20  # Use last 20 poses for scale estimation
        self.last_scale_update = 0

        # MINIMAL ADDITION: Enhanced scale information tracking
        self.scale_history = []
        self.scale_confidence = 0.0
        self.scale_stability_count = 0
        self.cumulative_visual_distance = 0.0
        self.cumulative_imu_distance = 0.0

        # Motion tracking
        self.motion_log = []
        self.stationary_count = 0

        # Camera intrinsic for Open3D
        height, width = config['camera']['height'], config['camera']['width']
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2]
        )

        print("Enhanced SLAM with detailed scale information initialized")

    def update_imu_trajectory(self, accel, gyro, dt):
        """IMU trajectory tracking with proper coordinate alignment"""
        if accel is None or dt <= 0 or dt > 0.1:
            return

        # APPLY COORDINATE ALIGNMENT TO IMU DATA
        # Transform acceleration from IMU frame to SLAM frame
        accel_4d = np.array([accel[0], accel[1], accel[2], 0])  # Homogeneous
        aligned_accel_4d = self.coordinate_aligner.camera_to_slam @ accel_4d
        aligned_accel = aligned_accel_4d[:3]  # Back to 3D

        # Remove gravity (simple approximation)
        accel_magnitude = np.linalg.norm(aligned_accel)
        if accel_magnitude > 5.0:  # Valid acceleration reading
            # Simple gravity removal - assume gravity is dominant component
            gravity_removed = max(0, accel_magnitude - 9.8)

            # Apply in direction of acceleration
            if accel_magnitude > 0:
                accel_direction = aligned_accel / accel_magnitude
                world_accel = accel_direction * gravity_removed
            else:
                world_accel = np.zeros(3)

            # Simple integration (with damping to prevent runaway)
            self.imu_velocity += world_accel * dt
            self.imu_velocity *= 0.95  # Damping factor

            prev_position = self.imu_position.copy()
            self.imu_position += self.imu_velocity * dt

            # MINIMAL ADDITION: Track cumulative IMU distance
            imu_step_distance = np.linalg.norm(self.imu_position - prev_position)
            self.cumulative_imu_distance += imu_step_distance

            # Store in trajectory
            self.imu_trajectory.append(self.imu_position.copy())
            self.trajectory_timestamps.append(time.time())

            # Keep trajectory manageable
            if len(self.imu_trajectory) > 1000:
                self.imu_trajectory.pop(0)
                self.trajectory_timestamps.pop(0)

    def estimate_scale_factor(self):
        """Enhanced scale estimation with confidence tracking"""
        if len(self.trajectory) < self.scale_estimation_window:
            return

        # Get recent trajectory segments
        visual_poses = self.trajectory[-self.scale_estimation_window:]
        imu_positions = self.imu_trajectory[-self.scale_estimation_window:]

        if len(imu_positions) < self.scale_estimation_window:
            return

        # Calculate distances traveled
        visual_distance = 0.0
        imu_distance = 0.0

        for i in range(1, len(visual_poses)):
            # Visual SLAM distance
            visual_pos1 = visual_poses[i - 1][:3, 3]
            visual_pos2 = visual_poses[i][:3, 3]
            visual_step = np.linalg.norm(visual_pos2 - visual_pos1)
            visual_distance += visual_step

            # MINIMAL ADDITION: Track cumulative visual distance
            self.cumulative_visual_distance += visual_step

            # IMU distance
            imu_pos1 = imu_positions[i - 1]
            imu_pos2 = imu_positions[i]
            imu_distance += np.linalg.norm(imu_pos2 - imu_pos1)

        # Estimate scale factor
        if visual_distance > 0.01 and imu_distance > 0.01:  # Minimum movement threshold
            new_scale = imu_distance / visual_distance

            # MINIMAL ADDITION: Track scale history and confidence
            self.scale_history.append(new_scale)
            if len(self.scale_history) > 10:
                self.scale_history.pop(0)

            # Calculate scale confidence based on stability
            if len(self.scale_history) >= 3:
                scale_variance = np.var(self.scale_history[-3:])
                self.scale_confidence = max(0.0, 1.0 - scale_variance * 10)

                # Track stability
                if scale_variance < 0.01:  # Low variance = stable
                    self.scale_stability_count += 1
                else:
                    self.scale_stability_count = 0

            # Simple moving average for stability
            alpha = 0.1  # Smoothing factor
            self.scale_factor = alpha * new_scale + (1 - alpha) * self.scale_factor

            print(f"Scale factor updated: {self.scale_factor:.3f} "
                  f"(IMU: {imu_distance:.3f}m, Visual: {visual_distance:.3f}m, "
                  f"Confidence: {self.scale_confidence:.2f})")

    # MINIMAL ADDITION: Enhanced scale information methods
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

    def get_aligned_trajectory(self):
        """Return visual trajectory aligned with IMU scale"""
        aligned_trajectory = []

        for pose in self.trajectory:
            aligned_pose = pose.copy()
            # Apply scale factor to translation component
            aligned_pose[:3, 3] *= self.scale_factor
            aligned_trajectory.append(aligned_pose)

        return aligned_trajectory

    def get_imu_trajectory_poses(self):
        """Convert IMU positions to 4x4 pose format for visualization"""
        imu_poses = []

        for position in self.imu_trajectory:
            pose = np.eye(4)
            pose[:3, 3] = position
            imu_poses.append(pose)

        return imu_poses

    def create_point_cloud(self, rgb, depth):
        """Create point cloud from RGB-D and align to SLAM frame"""
        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.params['max_depth'],
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

        if len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(self.params['voxel_size'])
            # Apply coordinate alignment
            pcd = self.coordinate_aligner.align_point_cloud(pcd)

        return pcd

    def estimate_pose(self, source_pcd, target_pcd):
        """ICP pose estimation"""
        if len(source_pcd.points) < 100 or len(target_pcd.points) < 100:
            return False, np.eye(4)

        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            self.params['icp_threshold'],
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=10
            )
        )

        return result.fitness > 0.1, result.transformation

    def process_frame(self, rgb, depth, accel=None, gyro=None):
        """Enhanced frame processing with motion detection and trajectory alignment"""
        self.frame_count += 1
        current_time = time.time()
        dt = current_time - self.last_time

        # Update IMU trajectory (MINIMAL ADDITION)
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

        # Only process SLAM if motion is detected
        if motion_result['decision']['should_update_slam']:
            self.process_slam_update(rgb, depth, motion_result)
        else:
            # If no motion, don't update pose but still count as stationary
            self.stationary_count += 1
            self.trajectory.append(self.current_pose.copy())

        # Update scale factor periodically (MINIMAL ADDITION)
        if self.frame_count - self.last_scale_update > 30:  # Every 30 frames
            self.estimate_scale_factor()
            self.last_scale_update = self.frame_count

        self.last_time = current_time

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

    def process_slam_update(self, rgb, depth, motion_result):
        """Process SLAM update when motion is detected"""
        current_pcd = self.create_point_cloud(rgb, depth)

        if len(current_pcd.points) == 0:
            return

        if self.prev_cloud is not None:
            success, transform = self.estimate_pose(current_pcd, self.prev_cloud)

            if success:
                transform_valid = self.validate_transform(transform, motion_result)

                if transform_valid:
                    # Apply coordinate alignment to pose transformation
                    aligned_transform = self.coordinate_aligner.align_pose(transform)
                    self.current_pose = np.dot(self.current_pose, aligned_transform)
                    self.trajectory.append(self.current_pose.copy())
                else:
                    self.trajectory.append(self.current_pose.copy())
            else:
                self.trajectory.append(self.current_pose.copy())

        # Accumulate to map less frequently
        if self.frame_count % self.params['accumulate_every_n'] == 0:
            self.accumulate_to_map(current_pcd)

        self.prev_cloud = current_pcd

    def validate_transform(self, transform, motion_result):
        """Validate transform against motion detection"""
        translation = np.linalg.norm(transform[:3, 3])
        rotation_angle = np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1))

        if motion_result['decision']['agreement'] == 'both_agree_motion':
            max_translation = 0.2
            max_rotation = 0.5
        elif motion_result['decision']['agreement'] == 'visual_only':
            max_translation = 0.1
            max_rotation = 0.3
        else:
            max_translation = 0.05
            max_rotation = 0.2

        return translation <= max_translation and rotation_angle <= max_rotation

    def accumulate_to_map(self, pcd):
        """Accumulate point cloud to map"""
        # Apply scale factor to map points for consistency
        scaled_pose = self.current_pose.copy()
        scaled_pose[:3, 3] *= self.scale_factor

        pcd_global = pcd.transform(scaled_pose)
        self.map_cloud += pcd_global

        if len(self.map_cloud.points) > self.params['max_points']:
            self.map_cloud = self.map_cloud.voxel_down_sample(self.params['voxel_size'] * 1.5)

            if len(self.map_cloud.points) > self.params['max_points']:
                indices = np.random.choice(
                    len(self.map_cloud.points),
                    self.params['max_points'],
                    replace=False
                )
                self.map_cloud = self.map_cloud.select_by_index(indices)

    def get_map(self):
        return self.map_cloud

    def get_trajectory(self):
        """Return scale-aligned visual trajectory"""
        return self.get_aligned_trajectory()

    def get_imu_trajectory(self):
        """Return IMU trajectory for visualization"""
        return self.get_imu_trajectory_poses()

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

        # MINIMAL ADDITION: Include detailed scale information
        scale_info = self.get_scale_info()

        return {
            'total_frames': total_frames,
            'motion_frames': motion_frames,
            'motion_rate': motion_frames / total_frames if total_frames > 0 else 0,
            'drift_detection_rate': drift_rate,
            'agreement_counts': agreement_counts,
            'scale_factor': self.scale_factor,
            'scale_info': scale_info  # Enhanced scale information
        }

    def save_session(self, filename):
        """Save session with comprehensive scale information"""
        o3d.io.write_point_cloud(f"{filename}_map.ply", self.map_cloud)

        # MINIMAL ADDITION: Enhanced session data with scale information
        scale_info = self.get_scale_info()

        session_data = {
            "visual_poses": [pose.tolist() for pose in self.get_aligned_trajectory()],
            "imu_poses": [pose.tolist() for pose in self.get_imu_trajectory()],
            "scale_factor": self.scale_factor,
            "scale_info": scale_info,  # Comprehensive scale information
            "frame_count": self.frame_count,
            "params": self.params,
            "motion_stats": self.get_motion_stats(),
            "motion_log": self.motion_log[-1000:]
        }

        with open(f"{filename}_trajectory.json", 'w') as f:
            json.dump(session_data, f, indent=2)

        with open(f"{filename}_motion_log.json", 'w') as f:
            json.dump(self.motion_log, f, indent=2)

        # MINIMAL ADDITION: Print detailed scale information
        print(f"Session saved with comprehensive scale information:")
        print(f"  Scale factor: {self.scale_factor:.3f}")
        print(f"  Scale confidence: {scale_info['scale_confidence']:.2f}")
        print(f"  Scale quality: {scale_info['scale_quality']}")
        print(f"  Total visual distance: {scale_info['cumulative_visual_distance']:.3f}m")
        print(f"  Total IMU distance: {scale_info['cumulative_imu_distance']:.3f}m")


def main():
    """Test enhanced SLAM with detailed scale information"""
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    from camera import D435iCamera

    print("Testing Enhanced SLAM with Detailed Scale Information...")
    camera = D435iCamera(config)
    intrinsics = camera.get_intrinsics()
    slam = EnhancedMinimalSLAM(intrinsics, config)

    try:
        for i in range(200):
            rgb, depth = camera.get_frames()
            accel, gyro = camera.get_imu_data()

            if rgb is not None and depth is not None:
                depth_meters = depth.astype(np.float32) / 1000.0
                slam.process_frame(rgb, depth_meters, accel, gyro)

                if i % 30 == 0:
                    stats = slam.get_motion_stats()
                    scale_info = stats.get('scale_info', {})
                    print(f"Frame {i}: "
                          f"Visual trajectory: {len(slam.get_trajectory())} poses, "
                          f"IMU trajectory: {len(slam.get_imu_trajectory())} poses, "
                          f"Scale factor: {stats.get('scale_factor', 1.0):.3f} "
                          f"({scale_info.get('scale_quality', 'Unknown')}), "
                          f"Map points: {len(slam.get_map().points)}")

        slam.save_session("../data/sessions/enhanced_scale_test")
        print("Enhanced SLAM with detailed scale information test complete")

    finally:
        camera.stop()


if __name__ == "__main__":
    main()