import cv2
import numpy as np
import time


class VisualIMUMotionDetector:
    def __init__(self):
        # IMU drift thresholds from your stationary test
        self.imu_thresholds = {
            'max_drift_rate': 0.17078,  # m/s from your test
            'position_threshold': 0.01,  # 1cm threshold for "no movement"
            'velocity_threshold': 0.02,  # 2cm/s threshold
            'gyro_threshold': 0.05  # rad/s threshold for rotation
        }

        # Visual motion detection
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Optical flow parameters
        self.flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # State tracking
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_corners = None

        # Motion analysis
        self.motion_history = []
        self.max_history = 10

        print("Visual-IMU Motion Detector initialized")
        print(f"IMU thresholds: {self.imu_thresholds}")

    def detect_visual_motion(self, current_frame):
        """Detect motion using feature matching and optical flow"""
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        visual_motion = {
            'has_motion': False,
            'feature_motion': 0.0,
            'flow_motion': 0.0,
            'feature_matches': 0,
            'method': 'none'
        }

        if self.prev_frame is None:
            self._initialize_tracking(gray)
            return visual_motion

        # Method 1: Feature matching
        feature_motion = self._detect_feature_motion(gray)

        # Method 2: Optical flow
        flow_motion = self._detect_optical_flow_motion(gray)

        # Combine results
        visual_motion['feature_motion'] = feature_motion['motion']
        visual_motion['flow_motion'] = flow_motion['motion']
        visual_motion['feature_matches'] = feature_motion['matches']

        # Determine if there's significant visual motion
        feature_threshold = 5.0  # pixels
        flow_threshold = 2.0  # pixels

        if feature_motion['motion'] > feature_threshold and feature_motion['matches'] > 20:
            visual_motion['has_motion'] = True
            visual_motion['method'] = 'features'
        elif flow_motion['motion'] > flow_threshold:
            visual_motion['has_motion'] = True
            visual_motion['method'] = 'optical_flow'

        # Update for next frame
        self.prev_frame = gray.copy()

        return visual_motion

    def _initialize_tracking(self, gray):
        """Initialize tracking features"""
        self.prev_frame = gray.copy()

        # Detect features for matching
        self.prev_keypoints, self.prev_descriptors = self.feature_detector.detectAndCompute(gray, None)

        # Detect corners for optical flow
        self.prev_corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7
        )

    def _detect_feature_motion(self, gray):
        """Detect motion using feature matching"""
        result = {'motion': 0.0, 'matches': 0}

        # Detect and match features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        if descriptors is not None and self.prev_descriptors is not None:
            matches = self.matcher.match(self.prev_descriptors, descriptors)

            if len(matches) > 10:
                # Calculate average displacement of matched features
                displacements = []
                for match in matches:
                    pt1 = self.prev_keypoints[match.queryIdx].pt
                    pt2 = keypoints[match.trainIdx].pt
                    displacement = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                    displacements.append(displacement)

                result['motion'] = np.mean(displacements)
                result['matches'] = len(matches)

        # Update for next frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return result

    def _detect_optical_flow_motion(self, gray):
        """Detect motion using optical flow"""
        result = {'motion': 0.0}

        if self.prev_corners is not None and len(self.prev_corners) > 0:
            # Calculate optical flow
            new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, self.prev_corners, None, **self.flow_params
            )

            # Select good points
            good_new = new_corners[status == 1]
            good_old = self.prev_corners[status == 1]

            if len(good_new) > 5:
                # Calculate average flow magnitude
                flow_vectors = good_new - good_old
                flow_magnitudes = np.sqrt(flow_vectors[:, 0] ** 2 + flow_vectors[:, 1] ** 2)
                result['motion'] = np.mean(flow_magnitudes)

        # Update corners for next frame
        self.prev_corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7
        )

        return result

    def analyze_imu_motion(self, accel, gyro, dt):
        """Analyze IMU motion and filter against known drift"""
        imu_motion = {
            'has_motion': False,
            'linear_motion': 0.0,
            'angular_motion': 0.0,
            'trusted': False
        }

        if accel is None or gyro is None:
            return imu_motion

        # Calculate motion magnitudes
        accel_mag = np.linalg.norm(accel)
        gyro_mag = np.linalg.norm(gyro)

        # Estimate linear motion (very rough)
        # Remove gravity first (assume roughly 9.8 in some direction)
        gravity_removed = max(0, accel_mag - 9.8)
        estimated_velocity_change = gravity_removed * dt

        imu_motion['linear_motion'] = estimated_velocity_change
        imu_motion['angular_motion'] = gyro_mag

        # Determine if IMU indicates motion (above drift thresholds)
        if estimated_velocity_change > self.imu_thresholds['velocity_threshold']:
            imu_motion['has_motion'] = True

        if gyro_mag > self.imu_thresholds['gyro_threshold']:
            imu_motion['has_motion'] = True

        # Trust IMU only if motion is significantly above noise threshold
        trust_threshold = 2.0  # 2x the drift rate
        if (estimated_velocity_change > self.imu_thresholds['velocity_threshold'] * trust_threshold or
                gyro_mag > self.imu_thresholds['gyro_threshold'] * trust_threshold):
            imu_motion['trusted'] = True

        return imu_motion

    def fuse_motion_detection(self, visual_motion, imu_motion, dt):
        """Fuse visual and IMU motion detection"""
        motion_decision = {
            'has_motion': False,
            'confidence': 0.0,
            'primary_source': 'none',
            'agreement': 'unknown',
            'should_update_slam': False
        }

        # Visual motion is always trusted (it's our ground truth)
        visual_says_motion = visual_motion['has_motion']
        imu_says_motion = imu_motion['has_motion']
        imu_trusted = imu_motion['trusted']

        # Determine agreement
        if visual_says_motion and imu_says_motion:
            motion_decision['agreement'] = 'both_agree_motion'
            motion_decision['has_motion'] = True
            motion_decision['confidence'] = 0.9
            motion_decision['primary_source'] = 'both'
            motion_decision['should_update_slam'] = True

        elif not visual_says_motion and not imu_says_motion:
            motion_decision['agreement'] = 'both_agree_stationary'
            motion_decision['has_motion'] = False
            motion_decision['confidence'] = 0.8
            motion_decision['primary_source'] = 'both'
            motion_decision['should_update_slam'] = False

        elif visual_says_motion and not imu_says_motion:
            motion_decision['agreement'] = 'visual_only'
            motion_decision['has_motion'] = True
            motion_decision['confidence'] = 0.7
            motion_decision['primary_source'] = 'visual'
            motion_decision['should_update_slam'] = True

        elif not visual_says_motion and imu_says_motion:
            if imu_trusted:
                motion_decision['agreement'] = 'imu_only_trusted'
                motion_decision['has_motion'] = True
                motion_decision['confidence'] = 0.4
                motion_decision['primary_source'] = 'imu'
                motion_decision['should_update_slam'] = True
            else:
                motion_decision['agreement'] = 'imu_drift_detected'
                motion_decision['has_motion'] = False  # Ignore IMU drift
                motion_decision['confidence'] = 0.8
                motion_decision['primary_source'] = 'visual'
                motion_decision['should_update_slam'] = False

        # Store in history
        self.motion_history.append({
            'timestamp': time.time(),
            'visual_motion': visual_motion['has_motion'],
            'imu_motion': imu_motion['has_motion'],
            'decision': motion_decision['has_motion'],
            'agreement': motion_decision['agreement']
        })

        # Keep history limited
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)

        return motion_decision

    def get_motion_stats(self):
        """Get motion detection statistics"""
        if len(self.motion_history) < 2:
            return {}

        recent_history = self.motion_history[-self.max_history:]

        agreement_counts = {}
        for entry in recent_history:
            agreement = entry['agreement']
            agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1

        total = len(recent_history)
        agreement_rates = {k: v / total for k, v in agreement_counts.items()}

        return {
            'total_samples': total,
            'agreement_rates': agreement_rates,
            'drift_detection_rate': agreement_rates.get('imu_drift_detected', 0.0)
        }


def main():
    """Test the motion detector"""
    from camera import D435iCamera

    print("Testing Visual-IMU Motion Detector")
    print("Move camera around to test motion detection")

    config = {"camera": {"width": 640, "height": 480, "fps": 30}}
    camera = D435iCamera(config)
    detector = VisualIMUMotionDetector()

    frame_count = 0
    last_time = time.time()

    try:
        while True:
            rgb, depth = camera.get_frames()
            accel, gyro = camera.get_imu_data()

            if rgb is None:
                continue

            current_time = time.time()
            dt = current_time - last_time

            # Detect motion
            visual_motion = detector.detect_visual_motion(rgb)
            imu_motion = detector.analyze_imu_motion(accel, gyro, dt)
            motion_decision = detector.fuse_motion_detection(visual_motion, imu_motion, dt)

            frame_count += 1

            # Print results every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}:")
                print(f"  Visual: {visual_motion['has_motion']} ({visual_motion['method']}, "
                      f"feat:{visual_motion['feature_motion']:.1f}px, flow:{visual_motion['flow_motion']:.1f}px)")
                print(f"  IMU: {imu_motion['has_motion']} (trusted:{imu_motion['trusted']}, "
                      f"lin:{imu_motion['linear_motion']:.4f}, ang:{imu_motion['angular_motion']:.4f})")
                print(f"  Decision: {motion_decision['has_motion']} ({motion_decision['agreement']}, "
                      f"conf:{motion_decision['confidence']:.1f}, update:{motion_decision['should_update_slam']})")

                stats = detector.get_motion_stats()
                if stats:
                    print(f"  Drift rate: {stats['drift_detection_rate']:.2%}")
                print()

            # Show frame
            cv2.imshow("Motion Detection Test", rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            last_time = current_time

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()