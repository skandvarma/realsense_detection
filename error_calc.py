import pyrealsense2 as rs
import numpy as np
import cv2

class EKF:
    #tried ekf but going with linear kalman filter
    def __init__(self, q=1.0, r=10.0, dt=1/30):#1/30 for 30 fps
        self.state = np.zeros((4,1))
        #x(k) = [x_pos(k), y_pos(k), x_vel(k), y_vel(k)]^T
        # x_pos(k + 1) = x_pos(k) + x_vel(k) * dt
        # y_pos(k + 1) = y_pos(k) + y_vel(k) * dt
        # x_vel(k + 1) = x_vel(k)  # Constant velocity assumption
        # y_vel(k + 1) = y_vel(k)  # Constant velocity assumption
        self.P = np.eye(4) * 1000
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]])
        #matrix form x(k+1) = F * x(k) + w(k)
        #measurement matrix
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])
        #noise convariance Q = G * G^T * σ_a²
        self.Q = np.array([[dt**4/4,0,dt**3/2,0],
                           [0,dt**4/4,0,dt**3/2],
                           [dt**3/2,0,dt**2,0],
                           [0,dt**3/2,0,dt**2]]) * q
        #measurement noise covariance
        self.R = np.eye(2) * r
        self.ready = False

    def init(self, x, y):
        self.state = np.array([[x],[y],[0],[0]])
        self.ready = True

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, m):
        z = np.array([[m[0]],[m[1]]])
        #previous tracking eqn
        y = z - self.H @ self.state
        #difference between actual measurement and predicted measurement
        S = self.H @ self.P @ self.H.T + self.R
        #K = P(k|k-1) * H^T * S^(-1) , K is the kalman gain a.) higher K = trust measurement more b.)Lower K = trust predict more
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get(self):
        return float(self.state[0]), float(self.state[1])


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
    align = rs.align(rs.stream.color)
    pipe.start(cfg)

    ekf = EKF(1.0,25.0)

    try:
        while True:
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            d = frames.get_depth_frame()
            if not d: continue

            img = np.asanyarray(d.get_data())
            my, mx = np.unravel_index(np.argmax(img), img.shape)

            size = 64
            h = size//2
            ys, ye = max(0,my-h), min(img.shape[0],my+h)
            xs, xe = max(0,mx-h), min(img.shape[1],mx+h)
            cl = img[ys:ye, xs:xe]

            rcx = xs + cl.shape[1]//2
            rcy = ys + cl.shape[0]//2

            if not ekf.ready:
                ekf.init(rcx, rcy)
                sx, sy = rcx, rcy
            else:
                ekf.predict()
                ekf.update([rcx, rcy])
                sx, sy = ekf.get()

            ex, ey = sx-320, sy-240
            dist = (ex**2+ey**2)**0.5
            maxd = img[my,mx]
            mean = np.mean(cl[cl>0])

            print(f"Max depth point: ({mx},{my}) = {maxd}mm")
            print(f"Raw cluster center: ({rcx},{rcy})")
            print(f"EKF smoothed center: ({sx:.1f},{sy:.1f})")
            print(f"Error: ({ex:.1f},{ey:.1f}), dist={dist:.2f}")
            print(f"Cluster mean depth: {mean:.2f}mm")
            print("-"*50)

            col = cv2.applyColorMap(cv2.convertScaleAbs(img,alpha=0.03), cv2.COLORMAP_JET)
            cv2.circle(col,(mx,my),2,(0,0,255),-1)
            cv2.circle(col,(int(rcx),int(rcy)),3,(0,255,255),-1)
            cv2.circle(col,(int(sx),int(sy)),3,(0,255,0),-1)

            cv2.rectangle(col,
                          (int(sx-h),int(sy-h)),
                          (int(sx+h),int(sy+h)),
                          (255,255,255),1)

            cv2.imshow("Depth", col)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
