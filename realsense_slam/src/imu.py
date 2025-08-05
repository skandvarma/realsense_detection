import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

# Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
pipeline.start(config)

# Setup data buffers
max_points = 100
times = deque(maxlen=max_points)
accel_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points), 'z': deque(maxlen=max_points)}
gyro_data = {'x': deque(maxlen=max_points), 'y': deque(maxlen=max_points), 'z': deque(maxlen=max_points)}

# Setup plots
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        accel = frames.first_or_default(rs.stream.accel)
        gyro = frames.first_or_default(rs.stream.gyro)

        if accel:
            a = accel.as_motion_frame().get_motion_data()
            current_time = time.time() - start_time

            times.append(current_time)
            accel_data['x'].append(a.x)
            accel_data['y'].append(a.y)
            accel_data['z'].append(a.z)

            print(f"Accel: x={a.x:.3f}, y={a.y:.3f}, z={a.z:.3f}")

        if gyro:
            g = gyro.as_motion_frame().get_motion_data()
            gyro_data['x'].append(g.x)
            gyro_data['y'].append(g.y)
            gyro_data['z'].append(g.z)

            print(f"Gyro:  x={g.x:.3f}, y={g.y:.3f}, z={g.z:.3f}")

        # Update plots every 10 frames
        if len(times) > 0 and len(times) % 10 == 0:
            # Clear and plot accelerometer
            ax1.clear()
            ax1.plot(times, accel_data['x'], 'r-', label='X')
            ax1.plot(times, accel_data['y'], 'g-', label='Y')
            ax1.plot(times, accel_data['z'], 'b-', label='Z')
            ax1.set_title('Accelerometer (m/s²)')
            ax1.legend()
            ax1.grid(True)

            # Clear and plot gyroscope
            ax2.clear()
            ax2.plot(times, gyro_data['x'], 'r-', label='X')
            ax2.plot(times, gyro_data['y'], 'g-', label='Y')
            ax2.plot(times, gyro_data['z'], 'b-', label='Z')
            ax2.set_title('Gyroscope (rad/s)')
            ax2.set_xlabel('Time (s)')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.pause(0.01)

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    plt.ioff()
    plt.show()