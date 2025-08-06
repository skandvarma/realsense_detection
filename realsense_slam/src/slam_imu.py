import numpy as np
import time


class IMUOnlySLAM:
    def __init__(self):
        # Pose tracking
        self.position = np.zeros(3)
        self.orientation = np.eye(3)
        self.velocity = np.zeros(3)

        # Trajectory storage
        self.trajectory = []
        self.timestamps = []

        # IMU bias estimation
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.bias_samples = []
        self.bias_estimated = False

        # State tracking
        self.last_time = None
        self.frame_count = 0

        # Drift reset mechanism
        self.last_reset_time = time.time()
        self.reset_interval = 5.0  # seconds
        self.reset_position = np.zeros(3)

        print("IMU-only SLAM initialized")
        print("Keep device stationary for first 3 seconds for calibration")

    def estimate_bias(self, accel, gyro):
        """Estimate sensor bias during initial stationary period"""
        if len(self.bias_samples) < 100:
            self.bias_samples.append([accel.copy(), gyro.copy()])
            return False

        if not self.bias_estimated:
            samples = np.array(self.bias_samples)
            gyro_samples = samples[:, 1]
            accel_samples = samples[:, 0]

            self.gyro_bias = np.mean(gyro_samples, axis=0)

            # Estimate gravity vector for initial orientation
            mean_accel = np.mean(accel_samples, axis=0)
            gravity_mag = np.linalg.norm(mean_accel)

            if gravity_mag > 5.0:
                gravity_dir = mean_accel / gravity_mag

                # Align device Z-axis with -gravity
                device_z = np.array([0, 0, 1])
                target_z = -gravity_dir

                # Rodrigues rotation formula for alignment
                v = np.cross(device_z, target_z)
                s = np.linalg.norm(v)
                c = np.dot(device_z, target_z)

                if s > 1e-6:
                    vx = np.array([[0, -v[2], v[1]],
                                   [v[2], 0, -v[0]],
                                   [-v[1], v[0], 0]])
                    self.orientation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))

                self.accel_bias = mean_accel - np.dot(self.orientation.T, np.array([0, 0, -9.81]))
                self.bias_estimated = True

                print(f"Bias estimation complete")
                print(f"Gyro bias: {self.gyro_bias}")

        return self.bias_estimated

    def update_orientation(self, gyro, dt):
        """Update orientation using gyroscope"""
        gyro_corrected = gyro - self.gyro_bias
        angular_velocity = np.linalg.norm(gyro_corrected)

        if angular_velocity > 0.01:
            axis = gyro_corrected / angular_velocity
            angle = angular_velocity * dt

            # Rodrigues rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])

            delta_R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            self.orientation = np.dot(self.orientation, delta_R)

    def update_position(self, accel, dt):
        """Update position using accelerometer"""
        accel_corrected = accel - self.accel_bias

        # Transform acceleration to world frame and remove gravity
        world_accel = np.dot(self.orientation, accel_corrected)
        world_accel[2] += 9.81  # Remove gravity

        # Integrate acceleration to get velocity and position
        self.velocity += world_accel * dt
        self.position += self.velocity * dt + 0.5 * world_accel * dt * dt

        # Apply damping to prevent drift
        self.velocity *= 0.98

    def reset_drift(self):
        """Reset position to prevent drift accumulation"""
        print(f"Resetting position drift at frame {self.frame_count}")
        self.reset_position = self.position.copy()
        self.velocity = np.zeros(3)
        self.last_reset_time = time.time()

    def process_imu_data(self, accel, gyro, timestamp=None):
        """Process IMU data to update pose"""
        if accel is None or gyro is None:
            return

        current_time = timestamp if timestamp else time.time()

        # Bias estimation phase
        if not self.bias_estimated:
            bias_ready = self.estimate_bias(accel, gyro)
            if not bias_ready:
                if self.frame_count % 30 == 0:
                    remaining = 100 - len(self.bias_samples)
                    print(f"Calibrating... {remaining} samples remaining")
                self.frame_count += 1
                return

        # Update pose if calibrated
        if self.last_time is not None:
            dt = current_time - self.last_time
            if 0 < dt < 0.1:  # Valid time step
                self.update_orientation(gyro, dt)
                self.update_position(accel, dt)

                # Store trajectory point
                pose = np.eye(4)
                pose[:3, :3] = self.orientation
                pose[:3, 3] = self.position - self.reset_position

                self.trajectory.append(pose)
                self.timestamps.append(current_time)

                # Check for drift reset
                if current_time - self.last_reset_time > self.reset_interval:
                    self.reset_drift()

        self.last_time = current_time
        self.frame_count += 1

    def get_trajectory(self):
        """Get current trajectory"""
        return self.trajectory

    def get_current_pose(self):
        """Get current pose matrix"""
        pose = np.eye(4)
        pose[:3, :3] = self.orientation
        pose[:3, 3] = self.position - self.reset_position
        return pose

    def save_trajectory(self, filename):
        """Save trajectory to file"""
        import json
        trajectory_data = {
            "poses": [pose.tolist() for pose in self.trajectory],
            "timestamps": self.timestamps
        }
        with open(f"{filename}_imu_trajectory.json", 'w') as f:
            json.dump(trajectory_data, f)