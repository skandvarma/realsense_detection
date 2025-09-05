#!/usr/bin/env python3
"""
ROS1 to ROS2 camera topic bridge using roslibpy
"""

import roslibpy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import base64
import numpy as np


class CameraBridge(Node):
    def __init__(self):
        super().__init__('camera_bridge')

        # ROS2 publishers
        self.color_pub = self.create_publisher(Image, '/camera/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/camera/aligned_depth_to_color/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera/color/camera_info', 10)

        # Connect to ROS1 via rosbridge
        self.ros1_client = roslibpy.Ros(host='192.168.1.189', port=9090)

        try:
            self.ros1_client.run()
            self.get_logger().info('Connected to ROS1 rosbridge')
            self.setup_subscribers()

        except Exception as e:
            self.get_logger().error(f'Failed to connect: {e}')

    def setup_subscribers(self):
        """Setup ROS1 topic subscribers"""

        # Color image
        self.color_sub = roslibpy.Topic(
            self.ros1_client,
            '/camera/color/image_raw',
            'sensor_msgs/Image'
        )
        self.color_sub.subscribe(self.color_callback)

        # Depth image
        self.depth_sub = roslibpy.Topic(
            self.ros1_client,
            '/camera/depth/image_rect_raw',
            'sensor_msgs/Image'
        )
        self.depth_sub.subscribe(self.depth_callback)

        # Camera info
        self.info_sub = roslibpy.Topic(
            self.ros1_client,
            '/camera/color/camera_info',
            'sensor_msgs/CameraInfo'
        )
        self.info_sub.subscribe(self.info_callback)

        self.get_logger().info('ROS1 subscribers setup complete')

    def color_callback(self, msg):
        try:
            ros2_msg = Image()
            ros2_msg.header = Header()
            ros2_msg.header.stamp = self.get_clock().now().to_msg()
            ros2_msg.header.frame_id = msg['header']['frame_id']
            ros2_msg.height = msg['height']
            ros2_msg.width = msg['width']
            ros2_msg.encoding = msg['encoding']
            ros2_msg.step = msg['step']
            ros2_msg.data = base64.b64decode(msg['data'])

            self.color_pub.publish(ros2_msg)

        except Exception as e:
            self.get_logger().error(f'Color callback error: {e}')

    def depth_callback(self, msg):
        try:
            ros2_msg = Image()
            ros2_msg.header = Header()
            ros2_msg.header.stamp = self.get_clock().now().to_msg()
            ros2_msg.header.frame_id = msg['header']['frame_id']
            ros2_msg.height = msg['height']
            ros2_msg.width = msg['width']
            ros2_msg.encoding = msg['encoding']
            ros2_msg.step = msg['step']
            ros2_msg.data = base64.b64decode(msg['data'])

            self.depth_pub.publish(ros2_msg)

        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')

    def info_callback(self, msg):
        try:
            ros2_msg = CameraInfo()
            ros2_msg.header = Header()
            ros2_msg.header.stamp = self.get_clock().now().to_msg()
            ros2_msg.header.frame_id = msg['header']['frame_id']
            ros2_msg.width = msg['width']
            ros2_msg.height = msg['height']
            ros2_msg.distortion_model = msg['distortion_model']
            ros2_msg.d = list(msg['D'])
            ros2_msg.k = list(msg['K'])
            ros2_msg.r = list(msg['R'])
            ros2_msg.p = list(msg['P'])

            self.info_pub.publish(ros2_msg)

        except Exception as e:
            self.get_logger().error(f'Info callback error: {e}')


def main():
    rclpy.init()
    bridge = CameraBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.ros1_client.close()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()