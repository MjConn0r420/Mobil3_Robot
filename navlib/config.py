"""All tunable constants in one place (SI units, radians)."""
import math

# ---------------- arena / grid ----------------
ARENA = 10.0            # arena is [0,ARENA]^2 metres
RES = 0.10              # occupancy-grid resolution [m/cell]
N = int(round(ARENA / RES))

# ---------------- robot (differential drive) ----------------
WHEEL_R = 0.05          # wheel radius [m]
TRACK = 0.26            # distance between wheels [m]
R_BODY = 0.18           # circumscribed radius of 0.30 x 0.20 m body [m]
R_COLLIDE = 0.15        # centre-to-obstacle distance regarded as a collision [m]
INFLATE = 0.25          # obstacle inflation used for planning (R_BODY + 7 cm margin) [m]
V_MAX = 0.50            # linear speed limit [m/s]
W_MAX = 2.0             # angular speed limit [rad/s]
WHEEL_MAX = 12.0        # wheel angular speed limit [rad/s]  (-> 0.6 m/s)
TAU_MOTOR = 0.12        # first-order motor lag [s]
DT = 0.032              # control step = Webots basicTimeStep (32 ms)
DELAY_STEPS = 2         # actuation / communication delay [steps]

# ---------------- pose measurement (GPS + compass in Webots) ----------------
POSE_SIGMA_XY = 0.01    # [m]
POSE_SIGMA_TH = 0.01    # [rad]

# ---------------- LiDAR (mounted at robot centre) ----------------
LIDAR_BEAMS = 360
LIDAR_RMIN = 0.05
LIDAR_RMAX = 5.0
LIDAR_HEIGHT = 0.20     # scan-plane height above floor [m]  (objects lower than this are INVISIBLE to it)
LIDAR_SIGMA = 0.01      # range noise std [m]
LIDAR_DROPOUT = 0.01    # probability that a beam returns nothing
LIDAR_SPIKE = 0.005     # probability of a random outlier return

# ---------------- camera (forward looking, horizontal optical axis) ----------------
CAM_W, CAM_H = 256, 128
CAM_FOV = 1.2           # horizontal field of view [rad]
CAM_F = (CAM_W / 2) / math.tan(CAM_FOV / 2)   # focal length [px]
CAM_HEIGHT = 0.25       # optical centre height above floor [m]
CAM_XOFF = 0.12         # camera forward offset from robot centre [m]
CAM_ZMAX = 3.5          # ignore ground-plane range estimates beyond this [m]
CAM_PIX_SIGMA = 6.0     # additive Gaussian pixel noise (0-255 scale)
CAM_SALT = 0.003        # fraction of salt & pepper pixels

# ---------------- occupancy grid (log-odds) ----------------
L_OCC_LIDAR, L_FREE_LIDAR = 0.90, -0.35
L_OCC_CAM, L_FREE_CAM = 0.60, -0.15
L_MIN, L_MAX = -2.0, 4.0
L_THRESH = 1.0          # cell is "occupied" when log-odds >= this (>= 2 agreeing observations)

# ---------------- task ----------------
START = (1.0, 1.0)
GOAL = (9.0, 9.0)
