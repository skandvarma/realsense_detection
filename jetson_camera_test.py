import cv2
import numpy as np
import time

class JetsonRealSenseCamera:
    """Simple camera interface for RealSense on Jetson with libuvc."""
    
    def __init__(self):
        self.color_cap = None
        self.depth_cap = None
        self.initialize()
    
    def initialize(self):
        """Initialize camera streams."""
        # Try different video device numbers
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Set resolution 
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test capture
                ret, frame = cap.read()
                if ret:
                    print(f"Found working camera at /dev/video{i}")
                    if self.color_cap is None:
                        self.color_cap = cap
                        print(f"   Using as color camera: {frame.shape}")
                    elif self.depth_cap is None and len(frame.shape) == 2:
                        self.depth_cap = cap
                        print(f"   Using as depth camera: {frame.shape}")
                    else:
                        cap.release()
                else:
                    cap.release()
    
    def get_frames(self):
        """Get color and simulated depth frames."""
        rgb = None
        depth = None
        
        if self.color_cap:
            ret, rgb = self.color_cap.read()
            if not ret:
                rgb = None
        
        # For now, create dummy depth (since UVC might not provide depth easily)
        if rgb is not None:
            # Create dummy depth frame for testing
            depth = np.random.randint(500, 2000, (rgb.shape[0], rgb.shape[1]), dtype=np.uint16)
            
        return rgb, depth
    
    def stop(self):
        """Release camera resources."""
        if self.color_cap:
            self.color_cap.release()
        if self.depth_cap:
            self.depth_cap.release()

# Test the camera
if __name__ == "__main__":
    print("Testing Jetson RealSense Camera...")
    
    camera = JetsonRealSenseCamera()
    
    for i in range(10):
        rgb, depth = camera.get_frames()
        
        if rgb is not None:
            print(f"Frame {i}: RGB {rgb.shape}, Depth {depth.shape if depth is not None else 'None'}")
        else:
            print(f"Frame {i}: No data")
        
        time.sleep(0.1)
    
    camera.stop()
    print("✅ Jetson camera test completed")
