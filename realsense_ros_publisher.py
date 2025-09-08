#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import asyncio
import threading
import cv2
import sys
import struct
import pickle

# Import your client
from EtherSenseClient import EtherSenseClient, ImageClient, multi_cast_message

class ROSImageClient(ImageClient):
    def __init__(self, reader, writer, source, ros_publisher):
        # Initialize parent class but override window creation
        self.reader = reader
        self.writer = writer
        self.address = writer.get_extra_info('peername')[0]
        self.port = source[1]
        self.depth_buffer = bytearray()
        self.color_buffer = bytearray()
        self.depth_remaining = 0
        self.color_remaining = 0
        self.frame_id = 0
        self.state = 'header'
        
        # Store ROS publisher reference instead of creating windows
        self.ros_publisher = ros_publisher
        
    def handle_frame(self):
        try:
            # Process depth frame
            depth_bytes = bytes(self.depth_buffer)
            depth_data = pickle.loads(depth_bytes, encoding='latin-1')
            
            if hasattr(depth_data, 'dtype') and depth_data.dtype != np.uint16:
                depth_data = depth_data.astype(np.uint16)
            
            # Process color frame
            color_bytes = bytes(self.color_buffer)
            color_data = pickle.loads(color_bytes, encoding='latin-1')
            
            # Publish to ROS topics instead of displaying
            self.ros_publisher.publish_frames(depth_data, color_data, self.timestamp)
            
            # Clear buffers
            self.depth_buffer = bytearray()
            self.color_buffer = bytearray()
            self.frame_id += 1
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            self.depth_buffer = bytearray()
            self.color_buffer = bytearray()

class ROSEtherSenseClient(EtherSenseClient):
    def __init__(self, ros_publisher):
        super().__init__()
        self.ros_publisher = ros_publisher
        
    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        print('Incoming connection from %s' % repr(addr))
        
        # Use ROS-enabled ImageClient
        handler = ROSImageClient(reader, writer, addr, self.ros_publisher)
        await handler.handle_connection()

class RealSenseROSPublisher(Node):
    def __init__(self):
        super().__init__('realsense_ros_publisher')
        
        # Publishers
        self.color_pub = self.create_publisher(
            Image, 
            '/camera/camera/color/image_raw', 
            10
        )
        self.depth_pub = self.create_publisher(
            Image, 
            '/camera/camera/aligned_depth_to_color/image_raw', 
            10
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, 
            '/camera/camera/color/camera_info', 
            10
        )
        
        # Parameters
        self.declare_parameter('mc_ip_address', '224.0.0.1')
        self.declare_parameter('port', 1024)
        
        self.bridge = CvBridge()
        
        # Camera intrinsics (will be populated when we get frames)
        self.fx = self.fy = 500.0
        self.ppx = self.ppy = 320.0
        self.width = 640
        self.height = 480
        
        self.get_logger().info('RealSense ROS Publisher node started')
        
        # Start the async client in a separate thread
        self.start_async_client()
    
    def start_async_client(self):
        def run_async_client():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            mc_ip = self.get_parameter('mc_ip_address').value
            port = self.get_parameter('port').value
            
            try:
                loop.run_until_complete(self.async_main(mc_ip, port))
            except Exception as e:
                self.get_logger().error(f'Async client error: {e}')
        
        self.client_thread = threading.Thread(target=run_async_client, daemon=True)
        self.client_thread.start()
    
    async def async_main(self, mc_ip, port):
        # Modified multicast client
        import socket
        
        multicast_group = (mc_ip, port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        try:
            self.get_logger().info(f'Sending discovery to {multicast_group}')
            sent = sock.sendto(b'EtherSensePing', multicast_group)
       
            # Start the ROS-enabled client
            client = ROSEtherSenseClient(self)
            await client.start_server()
            
        except Exception as e:
            self.get_logger().error(f'Client error: {e}')
        finally:
            sock.close()
    
    def publish_frames(self, depth_data, color_data, timestamp):
        try:
            now = self.get_clock().now().to_msg()
            
            # Publish color frame
            color_msg = self.bridge.cv2_to_imgmsg(color_data, encoding='bgr8')
            color_msg.header.stamp = now
            color_msg.header.frame_id = 'camera_color_optical_frame'
            self.color_pub.publish(color_msg)
            
            # Publish depth frame
            depth_msg = self.bridge.cv2_to_imgmsg(depth_data, encoding='16UC1')
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = 'camera_depth_optical_frame'
            self.depth_pub.publish(depth_msg)
            
            # Publish camera info
            self.publish_camera_info(now)
            
            # Update intrinsics from frame size if needed
            if hasattr(color_data, 'shape'):
                self.height, self.width = color_data.shape[:2]
                self.ppx = self.width / 2.0
                self.ppy = self.height / 2.0
            
        except Exception as e:
            self.get_logger().error(f'Publishing error: {e}')
    
    def publish_camera_info(self, timestamp):
        msg = CameraInfo()
        
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_color_optical_frame'
        
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = 'plumb_bob'
        
        # Intrinsic matrix K
        msg.k = [self.fx, 0.0, self.ppx,
                 0.0, self.fy, self.ppy,
                 0.0, 0.0, 1.0]
        
        # Rectification matrix R
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        
        # Projection matrix P
        msg.p = [self.fx, 0.0, self.ppx, 0.0,
                 0.0, self.fy, self.ppy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.camera_info_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RealSenseROSPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Shutting down...')
    except Exception as e:
        print(f'Node error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
