import open3d as o3d
import numpy as np
import json
import cv2
import time
from motion_detector import VisualIMUMotionDetector


class EnhancedMinimalSLAM:
    def __init__(self, intrinsics, config):
        self.intrinsics = intrinsics
        self.config = config

        # Initialize motion detector
        self.motion_detector = VisualIMUMotionDetector()

        # SLAM parameters (same as before)
        self.params = {
            'voxel_size': config.get('slam', {}).get('voxel_size', 0.05),
            'max_points': config.get('slam', {}).get('max_points', 20000),
            'icp_threshold': config.get('slam', {}).get('icp_threshold', 0.02),
            'max_depth': config.get('slam', {}).get('max_depth', 3.0),
            'process_every_n': config.get('slam', {}).get('process_every_n', 1),
            'accumulate_every_n': config.get('slam', {}).get('accumulate_every_n', 3)
        }

        # State
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        self.map_cloud = o3d.geometry.PointCloud()
        self.prev_cloud = None
        self.prev_rgb = None
        self.frame_count = 0
        self.last_time = time.time()

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

        print("Enhanced SLAM with Visual-IMU motion detection initialized")

    def create_point_cloud(self, rgb, depth):
        """Create point cloud from RGB-D (same as before)"""
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

        return pcd

    def estimate_pose(self, source_pcd, target_pcd):
        """ICP pose estimation (same as before)"""
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
        """Enhanced frame processing with motion detection"""
        self.frame_count += 1
        current_time = time.time()
        dt = current_time - self.last_time

        # Skip frames for performance if configured
        if self.frame_count % self.params['process_every_n'] != 0:
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
            self.trajectory.append(self.current_pose.copy())  # Same pose

        self.last_time = current_time

    def analyze_motion(self, rgb, accel, gyro, dt):
        """Analyze motion using visual-IMU detector"""
        # Visual motion detection
        visual_motion = self.motion_detector.detect_visual_motion(rgb)

        # IMU motion analysis
        imu_motion = self.motion_detector.analyze_imu_motion(accel, gyro, dt)

        # Fuse motion detection
        motion_decision = self.motion_detector.fuse_motion_detection(visual_motion, imu_motion, dt)

        return {
            'visual': visual_motion,
            'imu': imu_motion,
            'decision': motion_decision
        }

    def process_slam_update(self, rgb, depth, motion_result):
        """Process SLAM update when motion is detected"""
        # Create point cloud
        current_pcd = self.create_point_cloud(rgb, depth)

        if len(current_pcd.points) == 0:
            return

        # Pose estimation only if we have previous cloud
        if self.prev_cloud is not None:
            success, transform = self.estimate_pose(current_pcd, self.prev_cloud)

            if success:
                # Validate transform with motion detection confidence
                transform_valid = self.validate_transform(transform, motion_result)

                if transform_valid:
                    # Update pose
                    self.current_pose = np.dot(self.current_pose, transform)
                    self.trajectory.append(self.current_pose.copy())
                else:
                    # Reject transform, keep same pose
                    self.trajectory.append(self.current_pose.copy())
            else:
                # ICP failed, but motion was detected - keep same pose
                self.trajectory.append(self.current_pose.copy())

        # Accumulate to map less frequently
        if self.frame_count % self.params['accumulate_every_n'] == 0:
            self.accumulate_to_map(current_pcd)

        # Store for next frame
        self.prev_cloud = current_pcd

    def validate_transform(self, transform, motion_result):
        """Validate transform against motion detection"""
        # Extract translation and rotation from transform
        translation = np.linalg.norm(transform[:3, 3])
        rotation_angle = np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1))

        # Get motion confidence
        confidence = motion_result['decision']['confidence']

        # Validation thresholds based on motion type
        if motion_result['decision']['agreement'] == 'both_agree_motion':
            # High confidence - allow larger movements
            max_translation = 0.2  # 20cm
            max_rotation = 0.5  # ~30 degrees
        elif motion_result['decision']['agreement'] == 'visual_only':
            # Medium confidence - be more conservative
            max_translation = 0.1  # 10cm
            max_rotation = 0.3  # ~17 degrees
        else:
            # Low confidence - very conservative
            max_translation = 0.05  # 5cm
            max_rotation = 0.2  # ~11 degrees

        # Validate
        if translation > max_translation or rotation_angle > max_rotation:
            return False

        return True

    def accumulate_to_map(self, pcd):
        """Accumulate point cloud to map (same as before)"""
        pcd_global = pcd.transform(self.current_pose)
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
        return self.trajectory

    def get_motion_stats(self):
        """Get motion detection statistics"""
        if len(self.motion_log) < 10:
            return {}

        recent_log = self.motion_log[-100:]  # Last 100 frames

        total_frames = len(recent_log)
        motion_frames = sum(1 for entry in recent_log if entry['motion_detected'])
        stationary_frames = total_frames - motion_frames

        # Agreement analysis
        agreements = [entry['agreement'] for entry in recent_log]
        agreement_counts = {}
        for agreement in agreements:
            agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1

        # Drift detection rate
        drift_detections = sum(1 for entry in recent_log if entry['agreement'] == 'imu_drift_detected')
        drift_rate = drift_detections / total_frames if total_frames > 0 else 0

        return {
            'total_frames': total_frames,
            'motion_frames': motion_frames,
            'stationary_frames': stationary_frames,
            'motion_rate': motion_frames / total_frames if total_frames > 0 else 0,
            'drift_detection_rate': drift_rate,
            'agreement_counts': agreement_counts
        }

    def save_session(self, filename):
        """Save session with motion analysis"""
        # Save point cloud
        o3d.io.write_point_cloud(f"{filename}_map.ply", self.map_cloud)

        # Save trajectory and motion data
        session_data = {
            "poses": [pose.tolist() for pose in self.trajectory],
            "frame_count": self.frame_count,
            "params": self.params,
            "motion_stats": self.get_motion_stats(),
            "motion_log": self.motion_log[-1000:]  # Last 1000 entries
        }

        with open(f"{filename}_trajectory.json", 'w') as f:
            json.dump(session_data, f, indent=2)

        # Save motion log separately for analysis
        with open(f"{filename}_motion_log.json", 'w') as f:
            json.dump(self.motion_log, f, indent=2)


def main():
    """Test enhanced SLAM"""
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    from camera import D435iCamera

    print("Testing Enhanced SLAM with Motion Detection...")
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
                    print(f"Frame {i}: trajectory={len(slam.get_trajectory())}, "
                          f"map={len(slam.get_map().points)}, "
                          f"motion_rate={stats.get('motion_rate', 0):.2%}, "
                          f"drift_rate={stats.get('drift_detection_rate', 0):.2%}")

        slam.save_session("../data/sessions/enhanced_test")
        print("Enhanced SLAM test complete")

    finally:
        camera.stop()


if __name__ == "__main__":
    main()