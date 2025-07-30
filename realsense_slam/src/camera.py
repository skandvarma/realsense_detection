import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os


class D435iCamera:
    def __init__(self, config):
        self.config = config
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()

        # Configure streams
        cam_config = config["camera"]
        self.config_rs.enable_stream(rs.stream.depth, cam_config["width"], cam_config["height"], rs.format.z16,
                                     cam_config["fps"])
        self.config_rs.enable_stream(rs.stream.color, cam_config["width"], cam_config["height"], rs.format.bgr8,
                                     cam_config["fps"])

        # Start streaming
        self.profile = self.pipeline.start(self.config_rs)

        # Enable alignment
        self.align = rs.align(rs.stream.color)

        # Get intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def get_intrinsics(self):
        # Return intrinsic matrix
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return intrinsic_matrix

    def stop(self):
        self.pipeline.stop()


def main():
    # Test D435i camera functionality
    with open('../config/config.json', 'r') as f:
        config = json.load(f)

    camera = D435iCamera(config)
    print("Testing camera... Press 'q' to quit")

    try:
        for i in range(1000):
            rgb, depth = camera.get_frames()
            if rgb is not None and depth is not None:
                print(f"Frame {i}: RGB {rgb.shape}, Depth {depth.shape}")
                cv2.imshow('RGB', rgb)
                cv2.imshow('Depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
