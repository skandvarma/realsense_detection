import json
import time
import numpy as np
from camera import D435iCamera


class IMUDriftLogger:
    def __init__(self):
        # IMU state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)

        # Bias estimation
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.bias_samples = []
        self.bias_estimated = False

        # Logging
        self.start_time = time.time()
        self.last_time = None
        self.frame_count = 0

        # Data storage
        self.imu_data = {
            'positions': [],
            'timestamps': [],
            'frame_count': 0,
            'drift_analysis': {
                'max_drift': 0.0,
                'final_drift': 0.0,
                'drift_rate': 0.0,
                'avg_drift_per_second': 0.0,
                'incremental_drift_rate': 0.0,
                'drift_acceleration': 0.0
            }
        }

        # Drift tracking
        self.previous_drift = 0.0
        self.previous_time = 0.0
        self.drift_history = []

        print("IMU Drift Logger initialized")
        print("Keep device STATIONARY throughout the test")

    def estimate_bias(self, accel, gyro):
        """Simple bias estimation"""
        if len(self.bias_samples) < 100:
            self.bias_samples.append([accel.copy(), gyro.copy()])
            return False

        if not self.bias_estimated:
            samples = np.array(self.bias_samples)
            self.gyro_bias = np.mean(samples[:, 1], axis=0)

            # Simple accel bias (assuming stationary)
            mean_accel = np.mean(samples[:, 0], axis=0)
            gravity_mag = np.linalg.norm(mean_accel)

            if gravity_mag > 5.0:
                gravity_dir = mean_accel / gravity_mag
                # Assume Z should be -gravity
                expected_gravity = np.array([0, 0, -9.81])
                self.accel_bias = mean_accel - expected_gravity
                self.bias_estimated = True
                print(f"Bias estimated: gyro={self.gyro_bias}, accel_bias_mag={np.linalg.norm(self.accel_bias):.3f}")

        return self.bias_estimated

    def update_position(self, accel, gyro, dt):
        """Update position from IMU - minimal integration"""
        # Correct for bias
        accel_corrected = accel - self.accel_bias
        gyro_corrected = gyro - self.gyro_bias

        # Simple orientation update (just for gravity compensation)
        angular_speed = np.linalg.norm(gyro_corrected)
        if angular_speed > 0.01:
            axis = gyro_corrected / angular_speed
            angle = angular_speed * dt

            # Rodrigues rotation
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            delta_R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            self.orientation = np.dot(self.orientation, delta_R)

        # Transform acceleration and remove gravity
        world_accel = np.dot(self.orientation, accel_corrected)
        world_accel[2] += 9.81  # Remove gravity

        # Double integration
        self.velocity += world_accel * dt
        self.position += self.velocity * dt + 0.5 * world_accel * dt * dt

        # Apply damping to prevent unbounded drift
        self.velocity *= 0.99

    def log_position(self):
        """Log current position to data structure"""
        current_time = time.time() - self.start_time
        current_drift = np.linalg.norm(self.position)

        self.imu_data['positions'].append(self.position.tolist())
        self.imu_data['timestamps'].append(current_time)
        self.imu_data['frame_count'] = self.frame_count

        # Calculate drift metrics
        if len(self.imu_data['positions']) > 1:
            # Basic drift metrics
            self.imu_data['drift_analysis']['max_drift'] = max(
                self.imu_data['drift_analysis']['max_drift'],
                current_drift
            )
            self.imu_data['drift_analysis']['final_drift'] = current_drift

            # Average drift per second
            if current_time > 0:
                self.imu_data['drift_analysis']['avg_drift_per_second'] = current_drift / current_time

            # Incremental drift rate (drift change per time)
            if self.previous_time > 0:
                dt = current_time - self.previous_time
                if dt > 0:
                    drift_change = current_drift - self.previous_drift
                    incremental_rate = drift_change / dt
                    self.imu_data['drift_analysis']['incremental_drift_rate'] = incremental_rate

                    # Store drift history for acceleration calculation
                    self.drift_history.append({
                        'time': current_time,
                        'drift': current_drift,
                        'rate': incremental_rate
                    })

                    # Keep only recent history (last 10 points for smoothing)
                    if len(self.drift_history) > 10:
                        self.drift_history.pop(0)

                    # Calculate drift acceleration (how rate of drift is changing)
                    if len(self.drift_history) >= 3:
                        recent_rates = [h['rate'] for h in self.drift_history[-3:]]
                        rate_change = recent_rates[-1] - recent_rates[0]
                        time_span = self.drift_history[-1]['time'] - self.drift_history[-3]['time']
                        if time_span > 0:
                            self.imu_data['drift_analysis']['drift_acceleration'] = rate_change / time_span

        # Update tracking variables
        self.previous_drift = current_drift
        self.previous_time = current_time

    def save_to_file(self):
        """Save data to single JSON file"""
        with open('../data/imu_drift_log.json', 'w') as f:
            json.dump(self.imu_data, f, indent=2)

    def process_imu_data(self, accel, gyro):
        """Process single IMU sample"""
        if accel is None or gyro is None:
            return

        current_time = time.time()

        # Bias estimation phase
        if not self.bias_estimated:
            bias_ready = self.estimate_bias(accel, gyro)
            if not bias_ready:
                if self.frame_count % 30 == 0:
                    remaining = 100 - len(self.bias_samples)
                    print(f"Calibrating... {remaining} samples remaining")
                self.frame_count += 1
                return

        # Position update
        if self.last_time is not None and self.bias_estimated:
            dt = current_time - self.last_time
            if 0 < dt < 0.1:  # Valid time step
                self.update_position(accel, gyro, dt)
                self.log_position()

        self.last_time = current_time
        self.frame_count += 1

    def run_drift_test(self, duration_seconds=300):  # 5 minutes default
        """Run drift logging test"""
        print(f"Starting {duration_seconds}s drift test...")
        print("IMPORTANT: Keep device completely STATIONARY")

        # Initialize camera
        config = {"camera": {"width": 640, "height": 480, "fps": 30}}
        camera = D435iCamera(config)

        start_time = time.time()

        try:
            while time.time() - start_time < duration_seconds:
                # Get frames to trigger IMU update
                camera.get_frames()
                accel, gyro = camera.get_imu_data()

                # Process IMU
                self.process_imu_data(accel, gyro)

                # Save every 30 frames (about once per second)
                if self.frame_count % 30 == 0:
                    self.save_to_file()

                    if self.bias_estimated:
                        elapsed = time.time() - start_time
                        drift = np.linalg.norm(self.position)
                        avg_drift_rate = drift / elapsed if elapsed > 0 else 0

                        # Get incremental drift rate
                        incr_rate = self.imu_data['drift_analysis']['incremental_drift_rate']
                        drift_accel = self.imu_data['drift_analysis']['drift_acceleration']

                        print(f"Time: {elapsed:.1f}s | "
                              f"Position: [{self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f}] | "
                              f"Drift: {drift:.4f}m | "
                              f"Avg: {avg_drift_rate:.5f}m/s | "
                              f"Incr: {incr_rate:.5f}m/s | "
                              f"Accel: {drift_accel:.6f}m/s²")

                time.sleep(0.01)  # Small delay

        except KeyboardInterrupt:
            print("\nTest stopped by user")
        finally:
            camera.stop()
            self.save_to_file()
            self.print_final_analysis()

    def print_final_analysis(self):
        """Print drift analysis summary"""
        print("\n=== DRIFT ANALYSIS ===")
        data = self.imu_data['drift_analysis']
        print(f"Test duration: {self.imu_data['timestamps'][-1] if self.imu_data['timestamps'] else 0:.1f}s")
        print(f"Total frames: {self.imu_data['frame_count']}")
        print(f"Max drift: {data['max_drift']:.4f}m")
        print(f"Final drift: {data['final_drift']:.4f}m")
        print(f"Average drift per second: {data['avg_drift_per_second']:.5f}m/s")
        print(f"Incremental drift rate: {data['incremental_drift_rate']:.5f}m/s")
        print(f"Drift acceleration: {data['drift_acceleration']:.6f}m/s²")
        print(f"Final position: [{self.position[0]:.4f}, {self.position[1]:.4f}, {self.position[2]:.4f}]")

        # Drift characteristics analysis
        print("\n=== DRIFT CHARACTERISTICS ===")
        if data['drift_acceleration'] > 0.00001:
            print("Drift pattern: ACCELERATING (getting worse over time)")
        elif data['drift_acceleration'] < -0.00001:
            print("Drift pattern: DECELERATING (stabilizing over time)")
        else:
            print("Drift pattern: LINEAR (constant rate)")

        if data['avg_drift_per_second'] < 0.001:
            print("IMU Quality: EXCELLENT (< 1mm/s drift)")
        elif data['avg_drift_per_second'] < 0.005:
            print("IMU Quality: GOOD (< 5mm/s drift)")
        elif data['avg_drift_per_second'] < 0.01:
            print("IMU Quality: FAIR (< 10mm/s drift)")
        else:
            print("IMU Quality: POOR (> 10mm/s drift)")

        print(f"Data saved to: ../data/imu_drift_log.json")


def main():
    import os

    # Create data directory
    os.makedirs("../data", exist_ok=True)

    print("=== IMU Drift Logger ===")
    print("This tool logs IMU-derived positions to analyze drift")
    print("Device should be kept COMPLETELY STATIONARY")

    # Get test duration
    try:
        duration = int(input("Test duration in seconds (default 300): ") or "300")
    except ValueError:
        duration = 300

    logger = IMUDriftLogger()
    logger.run_drift_test(duration)


if __name__ == "__main__":
    main()