import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import time
import os


class RealSenseRTPBridgeNode(Node):
    def __init__(self):
        super().__init__('realsense_rtp_bridge')

        # Ensure GStreamer uses system plugins and reduce warning verbosity
        os.environ['GST_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/gstreamer-1.0/'
        os.environ['GST_DEBUG'] = '1'  # Reduce GStreamer log verbosity

        self.bridge = CvBridge()

        # Publishers for commonly used RealSense topics
        self.aligned_depth_pub = self.create_publisher(Image, '/camera/camera/aligned_depth_to_color/image_raw', 10)
        self.aligned_depth_info_pub = self.create_publisher(CameraInfo,
                                                            '/camera/camera/aligned_depth_to_color/camera_info', 10)

        self.color_pub = self.create_publisher(Image, '/camera/camera/color/image_raw', 10)
        self.color_info_pub = self.create_publisher(CameraInfo, '/camera/camera/color/camera_info', 10)

        self.depth_rect_pub = self.create_publisher(Image, '/camera/camera/depth/image_rect_raw', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/camera/camera/depth/camera_info', 10)

        self.points_pub = self.create_publisher(PointCloud2, '/camera/camera/depth/color/points', 10)

        # Match sender caps exactly and add proper buffering
        gst_pipeline = (
            'udpsrc port=5000 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            'rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw, format=BGR ! appsink drop=true sync=false max-buffers=1'
        )

        # Debug OpenCV GStreamer support first
        self.get_logger().info(f'OpenCV version: {cv2.__version__}')
        backends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]
        self.get_logger().info(f'Available backends: {backends}')

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        # Check if VideoCapture opened successfully with detailed logging
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open GStreamer pipeline')
            self.get_logger().error('OpenCV lacks GStreamer support - install python3-opencv system package')
            return
        else:
            self.get_logger().info('GStreamer pipeline opened successfully')

        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.timer_period = 1.0 / 25  # Reduced to 25 FPS for more stable reception
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.width = 1280  # set width matching your camera pipeline output
        self.height = 720  # set height matching your camera pipeline output
        self.frame_id = 'camera_link'

        self.camera_info = self.generate_dummy_camera_info()
        self.last_frame_time = time.time()
        self.first_frame_received = False

    def generate_dummy_camera_info(self):
        cam_info = CameraInfo()
        cam_info.width = self.width
        cam_info.height = self.height
        cam_info.distortion_model = 'plumb_bob'
        cam_info.d = [0., 0., 0., 0., 0.]

        cam_info.k = [600., 0., self.width / 2., 0., 600., self.height / 2., 0., 0., 1.]
        cam_info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        cam_info.p = [600., 0., self.width / 2., 0., 0., 600., self.height / 2., 0., 0., 0., 1., 0.]
        cam_info.header.frame_id = self.frame_id
        return cam_info

    def timer_callback(self):
        # Check if capture is still open
        if not self.cap.isOpened():
            self.get_logger().error('GStreamer pipeline closed, attempting reconnection')
            return

        ret, frame = self.cap.read()
        if not ret:
            current_time = time.time()
            if current_time - self.last_frame_time > 5.0:  # Log warning only every 5 seconds
                # Additional debugging info
                self.get_logger().warn('No frame received from GStreamer pipeline')
                self.get_logger().info(f'VideoCapture backend: {self.cap.getBackendName()}')
                self.get_logger().info(f'Pipeline is open: {self.cap.isOpened()}')
                self.last_frame_time = current_time
            return

        # Log first frame success immediately
        if not self.first_frame_received:
            self.get_logger().info(f'First frame received successfully: {frame.shape}')
            self.first_frame_received = True

        # Log successful frame reception periodically
        current_time = time.time()
        if current_time - self.last_frame_time > 5.0:  # Every 5 seconds
            self.get_logger().info(f'Successfully receiving frames: {frame.shape}')
        self.last_frame_time = current_time

        # Create color message
        ros_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        now = self.get_clock().now().to_msg()
        ros_image_msg.header.stamp = now
        ros_image_msg.header.frame_id = self.frame_id

        # Create dummy depth message (convert grayscale to 16UC1 depth format)
        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_dummy = gray.astype(np.uint16) * 10  # Scale to depth-like values
        depth_msg = self.bridge.cv2_to_imgmsg(depth_dummy, encoding='16UC1')
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = self.frame_id

        # Update camera_info timestamp
        self.camera_info.header.stamp = now

        # Publish with correct message types
        self.color_pub.publish(ros_image_msg)
        self.aligned_depth_pub.publish(depth_msg)  # Use proper depth format
        self.depth_rect_pub.publish(depth_msg)  # Use proper depth format

        self.color_info_pub.publish(self.camera_info)
        self.aligned_depth_info_pub.publish(self.camera_info)
        self.depth_info_pub.publish(self.camera_info)

        # Dummy PointCloud2 publish placeholder
        self.points_pub.publish(PointCloud2(header=ros_image_msg.header))


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRTPBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()