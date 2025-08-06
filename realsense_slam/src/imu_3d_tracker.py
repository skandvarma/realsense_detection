import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time


class IMULocalOrientationTracker:
    def __init__(self):
        # Setup RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.accel)
        self.config.enable_stream(rs.stream.gyro)

        # Initialize orientation relative to local reference
        self.initial_orientation = None  # Will be set as reference
        self.current_orientation = np.eye(3)  # Current absolute orientation
        self.relative_orientation = np.eye(3)  # Orientation relative to initial
        self.last_time = None

        # Bias estimation
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.bias_samples = []
        self.bias_estimated = False
        self.reference_set = False

        # Visualization setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("IMU Local Orientation Tracker", width=1000, height=700)

        # Create device representation at origin (local position = 0,0,0)
        self.device_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

        # Create RealSense device model
        self.device_body = o3d.geometry.TriangleMesh.create_box(0.15, 0.08, 0.03)
        self.device_body.translate([-0.075, -0.04, -0.015])
        self.device_body.paint_uniform_color([0.2, 0.2, 0.7])

        # Front indicators (lenses)
        self.lens1 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.005)
        lens1_transform = np.eye(4)
        lens1_transform[:3, 3] = [0.08, -0.02, 0]
        self.lens1.transform(lens1_transform)
        self.lens1.paint_uniform_color([0.1, 0.1, 0.1])

        self.lens2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.005)
        lens2_transform = np.eye(4)
        lens2_transform[:3, 3] = [0.08, 0.02, 0]
        self.lens2.transform(lens2_transform)
        self.lens2.paint_uniform_color([0.1, 0.1, 0.1])

        # Create reference frames
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.initial_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        self.initial_frame.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for initial reference

        # Create orientation history visualization
        self.orientation_history = []
        self.orientation_lines = o3d.geometry.LineSet()

        # Add geometries
        self.vis.add_geometry(self.world_frame)
        self.vis.add_geometry(self.initial_frame)
        self.vis.add_geometry(self.device_frame)
        self.vis.add_geometry(self.device_body)
        self.vis.add_geometry(self.lens1)
        self.vis.add_geometry(self.lens2)
        self.vis.add_geometry(self.orientation_lines)

        self.setup_camera_view()

        print("IMU Local Orientation Tracker")
        print("=============================")
        print("Device position is FIXED at origin (0,0,0)")
        print("Only orientation changes are tracked relative to initial pose")

    def setup_camera_view(self):
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([1, 1, 1])
        ctr.set_zoom(0.6)

    def estimate_bias(self, accel, gyro):
        """Estimate sensor bias during initial stationary period"""
        if len(self.bias_samples) < 100:
            self.bias_samples.append([accel, gyro])
            return False

        if not self.bias_estimated:
            samples = np.array(self.bias_samples)
            gyro_samples = samples[:, 1]
            accel_samples = samples[:, 0]

            # Estimate gyro bias
            self.gyro_bias = np.mean(gyro_samples, axis=0)

            # Estimate accel bias (remove gravity component)
            mean_accel = np.mean(accel_samples, axis=0)
            gravity_magnitude = np.linalg.norm(mean_accel)

            # Estimate initial gravity direction
            gravity_direction = mean_accel / gravity_magnitude

            # Calculate initial orientation from gravity
            # Assume device Z-axis should align with -gravity
            device_z = np.array([0, 0, 1])
            target_z = -gravity_direction

            # Calculate rotation to align device_z with target_z
            v = np.cross(device_z, target_z)
            s = np.linalg.norm(v)
            c = np.dot(device_z, target_z)

            if s > 1e-6:  # Not aligned
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                self.initial_orientation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
            else:
                self.initial_orientation = np.eye(3)

            self.current_orientation = self.initial_orientation.copy()
            self.bias_estimated = True
            self.reference_set = True

            print(f"Reference orientation set!")
            print(f"Gyro bias: [{self.gyro_bias[0]:.4f}, {self.gyro_bias[1]:.4f}, {self.gyro_bias[2]:.4f}]")
            print(f"Initial gravity: [{mean_accel[0]:.2f}, {mean_accel[1]:.2f}, {mean_accel[2]:.2f}]")

        return True

    def update_orientation(self, gyro, dt):
        """Update orientation using bias-corrected gyroscope"""
        gyro_corrected = gyro - self.gyro_bias
        angular_speed = np.linalg.norm(gyro_corrected)

        if angular_speed > 0.01:  # Threshold to avoid noise
            axis = gyro_corrected / angular_speed
            angle = angular_speed * dt

            # Rodrigues' rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])

            delta_R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            self.current_orientation = np.dot(self.current_orientation, delta_R)

            # Calculate relative orientation from initial reference
            self.relative_orientation = np.dot(self.current_orientation, self.initial_orientation.T)

    def correct_orientation_with_accel(self, accel, alpha=0.02):
        """Correct orientation drift using accelerometer"""
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            accel_unit = accel / accel_norm

            # Expected gravity direction in current device frame
            gravity_device = np.array([0, 0, -1])

            # Current gravity direction in world frame
            gravity_world = np.dot(self.current_orientation, gravity_device)

            # Correction to align with measured gravity
            cross_product = np.cross(gravity_world, accel_unit)
            sin_angle = np.linalg.norm(cross_product)
            cos_angle = np.dot(gravity_world, accel_unit)

            if sin_angle > 0.01:
                axis = cross_product / sin_angle
                correction_angle = alpha * np.arctan2(sin_angle, cos_angle)

                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])

                correction_R = np.eye(3) + np.sin(correction_angle) * K + \
                               (1 - np.cos(correction_angle)) * np.dot(K, K)

                self.current_orientation = np.dot(correction_R, self.current_orientation)
                self.relative_orientation = np.dot(self.current_orientation, self.initial_orientation.T)

    def update_visualization(self):
        """Update 3D visualization - device stays at origin, only rotates"""
        # Create transformation matrix (translation = 0, only rotation)
        transform = np.eye(4)
        transform[:3, :3] = self.relative_orientation
        transform[:3, 3] = [0, 0, 0]  # Always at origin

        # Update device geometries
        self.device_frame.transform(transform)
        self.device_body.transform(transform)
        self.lens1.transform(transform)
        self.lens2.transform(transform)

        # Store orientation history for visualization
        # Extract forward vector (X-axis) from current orientation
        forward_vector = self.relative_orientation[:, 0] * 0.2  # Scale for visualization
        self.orientation_history.append(forward_vector.copy())

        # Limit history to prevent memory issues
        if len(self.orientation_history) > 100:
            self.orientation_history.pop(0)

        # Create lines showing orientation history
        if len(self.orientation_history) > 1:
            points = [[0, 0, 0]]  # Start from origin
            for vec in self.orientation_history:
                points.append(vec)

            lines = [[i, i + 1] for i in range(len(points) - 1)]

            self.orientation_lines.points = o3d.utility.Vector3dVector(points)
            self.orientation_lines.lines = o3d.utility.Vector2iVector(lines)
            colors = [[1, 0.5, 0] for _ in range(len(lines))]  # Orange trail
            self.orientation_lines.colors = o3d.utility.Vector3dVector(colors)

        # Update geometries
        self.vis.update_geometry(self.device_frame)
        self.vis.update_geometry(self.device_body)
        self.vis.update_geometry(self.lens1)
        self.vis.update_geometry(self.lens2)
        self.vis.update_geometry(self.orientation_lines)

        # Reset geometries for next frame
        self.device_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

        self.device_body = o3d.geometry.TriangleMesh.create_box(0.15, 0.08, 0.03)
        self.device_body.translate([-0.075, -0.04, -0.015])
        self.device_body.paint_uniform_color([0.2, 0.2, 0.7])

        self.lens1 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.005)
        lens1_transform = np.eye(4)
        lens1_transform[:3, 3] = [0.08, -0.02, 0]
        self.lens1.transform(lens1_transform)
        self.lens1.paint_uniform_color([0.1, 0.1, 0.1])

        self.lens2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.005)
        lens2_transform = np.eye(4)
        lens2_transform[:3, 3] = [0.08, 0.02, 0]
        self.lens2.transform(lens2_transform)
        self.lens2.paint_uniform_color([0.1, 0.1, 0.1])

    def get_euler_angles(self):
        """Convert relative orientation to Euler angles"""
        R = self.relative_orientation
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.degrees([x, y, z])  # Roll, Pitch, Yaw in degrees

    def reset_reference(self):
        """Reset current orientation as new reference"""
        self.initial_orientation = self.current_orientation.copy()
        self.relative_orientation = np.eye(3)
        self.orientation_history = []
        print("Reference orientation reset to current pose")

    def run(self):
        print("\nInstructions:")
        print("1. Keep device STATIONARY for first 3 seconds")
        print("2. Device will stay at origin (0,0,0)")
        print("3. Rotate device to see orientation changes")
        print("4. Gray frame = initial reference orientation")
        print("5. Colored frame = current device orientation")
        print("6. Orange trail = orientation history")
        print("\nPress 'R' to reset reference or close window to exit")

        try:
            self.pipeline.start(self.config)
            frame_count = 0

            while True:
                frames = self.pipeline.wait_for_frames()
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)

                current_time = time.time()

                if accel_frame and gyro_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()

                    accel = np.array([accel_data.x, accel_data.y, accel_data.z])
                    gyro = np.array([gyro_data.x, gyro_data.y, gyro_data.z])

                    # Estimate bias and set reference
                    if not self.bias_estimated:
                        bias_ready = self.estimate_bias(accel, gyro)
                        if not bias_ready:
                            if frame_count % 30 == 0:
                                remaining = 100 - len(self.bias_samples)
                                print(f"Calibrating... {remaining} samples remaining (keep stationary)")
                            frame_count += 1
                            continue

                    if self.last_time is not None and self.reference_set:
                        dt = current_time - self.last_time
                        if 0.001 < dt < 0.1:
                            # Update orientation
                            self.update_orientation(gyro, dt)

                            # Correct with accelerometer
                            self.correct_orientation_with_accel(accel)

                            # Update visualization
                            if frame_count % 2 == 0:
                                self.update_visualization()

                    self.last_time = current_time
                    frame_count += 1

                    # Print orientation every 30 frames
                    if frame_count % 30 == 0 and self.reference_set:
                        roll, pitch, yaw = self.get_euler_angles()
                        print(f"Frame {frame_count}: Roll: {roll:6.1f}°, Pitch: {pitch:6.1f}°, Yaw: {yaw:6.1f}°")

                # Handle visualization events
                if not self.vis.poll_events():
                    break

                # Check for reset (simplified - reset every 1000 frames if needed)
                if frame_count % 100000 == 0 and frame_count > 0 and self.reference_set:
                    print("Auto-reset reference to prevent long-term drift")
                    self.reset_reference()

                self.vis.update_renderer()

        except KeyboardInterrupt:
            print("\nIMU orientation tracker stopped")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.pipeline.stop()
            self.vis.destroy_window()


def main():
    tracker = IMULocalOrientationTracker()
    tracker.run()


if __name__ == "__main__":
    main()