import numpy as np
import cv2
import zmq
import json
import time
from typing import Tuple
from dataclasses import dataclass

# sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera.realsense_manager import CameraShareManager

def get_detection_flag():
    """Read detection status from file."""
    try:
        with open("detection_flag.txt", "r") as f:
            return f.read().strip() == "1"
    except:
        return False

def set_detection_flag(value: bool):
    """Write detection status to file."""
    try:
        with open("detection_flag.txt", "w") as f:
            f.write("1" if value else "0")
    except:
        pass

def analyze_frame_occupancy(depth_frame, threshold_mm=800, occupancy_threshold=0.7):
    """
    Analyze frame occupancy based on depth data.
    Returns True if obstacle covers >= occupancy_threshold of center region.
    """
    height, width = depth_frame.shape

    # Define center region (40% of frame like in C++ code)
    center_x_start = int(width * 0.3)
    center_x_end = int(width * 0.7)
    center_y_start = int(height * 0.3)
    center_y_end = int(height * 0.7)

    # Extract center region
    center_region = depth_frame[center_y_start:center_y_end, center_x_start:center_x_end]

    # Count valid pixels (depth > 0)
    valid_pixels = np.count_nonzero(center_region)

    if valid_pixels == 0:
        return False

    # Count close pixels (within threshold distance)
    close_pixels = np.count_nonzero((center_region > 0) & (center_region <= threshold_mm))

    # Calculate occupancy percentage
    occupancy_percentage = close_pixels / valid_pixels

    return occupancy_percentage >= occupancy_threshold

@dataclass
class PIDState:
    """State information for PID controller."""
    last_error_x: float = 0.0
    last_error_y: float = 0.0
    last_time: float = 0.0
    initialized: bool = False


class PDController:
    """PD controller for centering detections."""

    def __init__(self, p_gain: float = -0.0008, d_gain: float = 0.00010,
                 max_velocity: float = 3.0, min_velocity: float = 0.001):
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.state = PIDState()
        self.frame_center_x = 320  # 640x480 default
        self.frame_center_y = 240

    def set_frame_dimensions(self, width: int, height: int) -> None:
        self.frame_center_x = width // 2
        self.frame_center_y = height // 2

    def compute(self, detection_x: int, detection_y: int, dt: float) -> Tuple[float, float]:
        error_x = detection_x - self.frame_center_x
        error_y = detection_y - self.frame_center_y

        current_time = time.time()

        if not self.state.initialized:
            self.state.last_error_x = error_x
            self.state.last_error_y = error_y
            self.state.last_time = current_time
            self.state.initialized = True
            return 0.0, 0.0

        if dt > 0:
            derivative_x = (error_x - self.state.last_error_x) / dt
            derivative_y = (error_y - self.state.last_error_y) / dt
        else:
            derivative_x = 0.0
            derivative_y = 0.0

        velocity_x = -(self.p_gain * error_x + self.d_gain * derivative_x)
        velocity_y = -(self.p_gain * error_y + self.d_gain * derivative_y)

        velocity_x = self._apply_limits(velocity_x)
        velocity_y = self._apply_limits(velocity_y)

        self.state.last_error_x = error_x
        self.state.last_error_y = error_y
        self.state.last_time = current_time

        return velocity_x, velocity_y

    def _apply_limits(self, velocity: float) -> float:
        if abs(velocity) < self.min_velocity:
            return 0.0
        if velocity > self.max_velocity:
            return self.max_velocity
        elif velocity < -self.max_velocity:
            return -self.max_velocity
        return velocity

    def reset(self) -> None:
        self.state = PIDState()


class EKF:
    def __init__(self, q=1.0, r=10.0, dt=1 / 30):
        self.state = np.zeros((4, 1))
        self.P = np.eye(4) * 1000
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                           [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                           [dt ** 3 / 2, 0, dt ** 2, 0],
                           [0, dt ** 3 / 2, 0, dt ** 2]]) * q
        self.R = np.eye(2) * r
        self.ready = False

    def init(self, x, y):
        self.state = np.array([[x], [y], [0], [0]])
        self.ready = True

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, m):
        z = np.array([[m[0]], [m[1]]])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get(self):
        return float(self.state[0]), float(self.state[1])


class UnitreeDepthController:
    def __init__(self, zmq_address="tcp://192.168.1.162:5555", config_path="config.yaml"):
        # Initialize ROS2 camera system (same as test_detection.py)
        self.camera_manager = CameraShareManager()
        self.subscriber_id = None

        # Load configuration (minimal config for camera)
        self.config = self._load_config(config_path)

        # Initialize camera
        if not self._initialize_camera():
            raise RuntimeError("Failed to initialize ROS2 camera")

        # Initialize EKF
        self.ekf = EKF(1.0, 25.0)

        # Initialize PD Controller
        self.pd_controller = PDController(
            p_gain=-0.0008,
            d_gain=0.00010,
            max_velocity=1.0,
            min_velocity=0.01
        )
        self.pd_controller.set_frame_dimensions(640, 480)

        # Initialize ZMQ publisher
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(zmq_address)
        print(f"ZMQ publisher bound to {zmq_address}")

        # Timing
        self.last_time = time.time()

    def _load_config(self, config_path):
        """Load minimal configuration for camera."""
        # Minimal config if file doesn't exist
        default_config = {
            'ros2': {
                'color_topic': '/camera/camera/color/image_raw',
                'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/camera/camera/color/camera_info'
            }
        }

        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
            return default_config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return default_config

    def _initialize_camera(self):
        """Initialize ROS2 camera system (same logic as test_detection.py)."""
        try:
            print("Initializing ROS2 camera system...")

            if self.camera_manager.initialize_camera(self.config):
                self.subscriber_id = self.camera_manager.register_subscriber("DepthController")
                print(f"Registered as ROS2 subscriber: {self.subscriber_id}")

                # Wait for camera to start streaming
                print("Waiting for ROS2 camera frames...")
                wait_start = time.time()
                while time.time() - wait_start < 10.0:
                    color_frame, depth_frame = self.camera_manager.get_frames_for_subscriber(self.subscriber_id)
                    if color_frame is not None:
                        print("ROS2 frames received, camera ready")
                        return True
                    time.sleep(0.1)

                print("No frames received from ROS2 camera within 10 seconds")
                return False
            else:
                print("ROS2 camera initialization failed")
                return False

        except Exception as e:
            print(f"ROS2 camera setup failed: {e}")
            print("Make sure ROS2 RealSense node is running:")
            print("  ros2 launch realsense2_camera rs_launch.py")
            return False

    def start(self):
        """Start the controller (camera already initialized)."""
        print("ROS2 camera system ready")

    def stop(self):
        """Stop the controller and cleanup."""
        if self.subscriber_id:
            self.camera_manager.unregister_subscriber(self.subscriber_id)
        self.socket.close()
        self.context.term()
        cv2.destroyAllWindows()
        print("Cleanup complete")

    def process_frame(self):
        """Process frame using ROS2 camera data."""
        # Get frames from ROS2 subscriber (same as test_detection.py)
        color_frame, depth_frame = self.camera_manager.get_frames_for_subscriber(self.subscriber_id)

        if color_frame is None or depth_frame is None:
            return None

        # Convert depth frame to numpy array (ROS2 frames are already numpy arrays)
        img = depth_frame

        # Always show visualization - initialize default values
        col = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.03), cv2.COLORMAP_JET)
        data = None
        valid_detection = False

        h_start, h_end = 210, 270
        sub_img = img[h_start:h_end, :]

        if sub_img.size > 0:
            min_width, min_height = 19, 14

            # Get depth threshold (80% of max depth in region)
            max_depth_val = np.max(sub_img)
            if max_depth_val > 0:
                depth_threshold = max_depth_val * 0.8

                # Create binary mask of significant depth values
                mask = (sub_img >= depth_threshold) & (sub_img > 0)

                # Find connected components using simple labeling
                labeled = np.zeros_like(mask, dtype=int)
                component_id = 0
                visited = np.zeros_like(mask, dtype=bool)

                def flood_fill(start_y, start_x, comp_id):
                    stack = [(start_y, start_x)]
                    pixels = []
                    while stack:
                        y, x = stack.pop()
                        if (y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1] or
                                visited[y, x] or not mask[y, x]):
                            continue
                        visited[y, x] = True
                        labeled[y, x] = comp_id
                        pixels.append((y, x))
                        stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
                    return pixels

                # Find all connected components
                components = []
                for y in range(mask.shape[0]):
                    for x in range(mask.shape[1]):
                        if mask[y, x] and not visited[y, x]:
                            component_id += 1
                            pixels = flood_fill(y, x, component_id)
                            if pixels:
                                components.append(pixels)

                # Find largest component that meets size requirements
                best_component = None
                best_size = 0

                for pixels in components:
                    if len(pixels) == 0:
                        continue

                    # Get bounding box of component
                    ys = [p[0] for p in pixels]
                    xs = [p[1] for p in pixels]
                    y_min, y_max = min(ys), max(ys)
                    x_min, x_max = min(xs), max(xs)
                    comp_width = x_max - x_min + 1
                    comp_height = y_max - y_min + 1

                    # Check if component meets size requirements
                    if comp_width >= min_width and comp_height >= min_height:
                        if len(pixels) > best_size:
                            best_component = pixels
                            best_size = len(pixels)

                # Only proceed with tracking if valid object found
                if best_component is not None:
                    valid_detection = True

                    # Get center and bounds of best component
                    ys = [p[0] for p in best_component]
                    xs = [p[1] for p in best_component]
                    y_min, y_max = min(ys), max(ys)
                    x_min, x_max = min(xs), max(xs)

                    # Convert back to full image coordinates
                    sub_my, sub_mx = np.mean(ys), np.mean(xs)
                    my, mx = int(h_start + sub_my), int(sub_mx)

                    # Extract actual component region for analysis
                    ys, ye = h_start + y_min, h_start + y_max + 1
                    xs, xe = x_min, x_max + 1
                    cl = img[ys:ye, xs:xe]

                    # Calculate raw cluster center
                    rcx = xs + cl.shape[1] // 2
                    rcy = ys + cl.shape[0] // 2

                    # Apply EKF filtering
                    if not self.ekf.ready:
                        self.ekf.init(rcx, rcy)
                        sx, sy = rcx, rcy
                    else:
                        self.ekf.predict()
                        self.ekf.update([rcx, rcy])
                        sx, sy = self.ekf.get()

                    # Calculate time delta
                    current_time = time.time()
                    dt = current_time - self.last_time
                    self.last_time = current_time

                    # Compute velocity commands using PD controller
                    vx, vy = self.pd_controller.compute(int(sx), int(sy), dt)

                    # Prepare data for transmission
                    data = {
                        "timestamp": current_time,
                        "vx": float(vx),
                        "vy": float(vy),
                        "target_x": int(sx),
                        "target_y": int(sy),
                        "error_x": float(sx - 320),
                        "error_y": float(sy - 240),
                        "max_depth": int(img[my, mx]) if img[my, mx] > 0 else 0,
                        "mean_depth": float(np.mean(cl[cl > 0])) if np.any(cl > 0) else 0.0
                    }

                    # Draw detection visualization
                    cv2.circle(col, (mx, my), 2, (0, 0, 255), -1)
                    cv2.circle(col, (int(rcx), int(rcy)), 3, (0, 255, 255), -1)
                    cv2.circle(col, (int(sx), int(sy)), 3, (0, 255, 0), -1)
                    cv2.rectangle(col, (xs, ys), (xe, ye), (255, 255, 255), 1)

                    # Add text overlay
                    cv2.putText(col, f"Vx: {vx:.3f} m/s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(col, f"Vy: {vy:.3f} m/s", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        obstacle_detected = analyze_frame_occupancy(img, threshold_mm=800, occupancy_threshold=0.7)
        object_detected_flag = get_detection_flag()

        # Send appropriate command based on detection type
        if obstacle_detected:
            current_time = time.time()
            obs_stop_data = {
                "command": "obs_stop",
                "timestamp": current_time
            }
            message = json.dumps(obs_stop_data)
            self.socket.send_string(message)
            print("obs_stop sent - depth obstacle")
        elif object_detected_flag:
            current_time = time.time()
            stop_data = {
                "command": "stop",
                "timestamp": current_time
            }
            message = json.dumps(stop_data)
            self.socket.send_string(message)
            print("stop sent - object detected")
        elif valid_detection and data:
            message = json.dumps(data)
            self.socket.send_string(message)

        # Always show status
        if not valid_detection:
            cv2.putText(col, "No valid target", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add ROS2 info
        cv2.putText(col, "ROS2 Camera", (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Unitree A1 Depth Tracking", col)

        return data

    def run(self):
        print("Starting Unitree A1 depth tracking controller with ROS2 camera...")
        print("Make sure ROS2 RealSense node is running:")
        print("  ros2 launch realsense2_camera rs_launch.py")
        print("Press 'q' to quit")

        self.start()

        try:
            while True:
                data = self.process_frame()

                if data:
                    print(f"Vx: {data['vx']:.3f} m/s, Vy: {data['vy']:.3f} m/s, "
                          f"Error: ({data['error_x']:.1f}, {data['error_y']:.1f}), "
                          f""
                          f"Depth: {data['max_depth']}mm")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()


def main():
    zmq_address = "tcp://192.168.1.162:5555"
    config_path = "config.yaml"

    controller = UnitreeDepthController(zmq_address, config_path)
    controller.run()


if __name__ == "__main__":
    main()