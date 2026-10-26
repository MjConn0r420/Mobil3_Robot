"""Minimal stand-in for Webots' `controller` module, backed by the offline simulator.
It deliberately uses DIFFERENT conventions from navlib (reversed + shifted LiDAR beam order, flipped and
offset IMU yaw) so the controller's start-up calibration is exercised."""
import math
import numpy as np
from navlib import config as C
from navlib.world import World
from navlib.sim import RobotSim
from navlib.sensors import SimLidar, SimCamera

BEAM_SHIFT = 37      # raw[j] = canonical[(BEAM_SHIFT - j) % N]
YAW_SIGN, YAW_OFF = -1.0, 0.7
T_MAX = 600.0


class _Motor:
    def __init__(self): self.v = 0.0
    def setPosition(self, p): pass
    def setVelocity(self, v): self.v = v


class _Dev:
    def __init__(self, robot, kind): self.r, self.kind = robot, kind
    def enable(self, ts): pass
    def getValues(self):
        return [self.r.sim.x, self.r.sim.y, 0.0]          # Webots GPS is noise-free unless `noise` is set
    def getRollPitchYaw(self):
        th = self.r.sim.th
        return [0.0, 0.0, (YAW_SIGN * th + YAW_OFF + math.pi) % (2 * math.pi) - math.pi]
    def getRangeImage(self):
        canon = self.r.lidar.scan(self.r.sim.true_pose())
        n = len(canon)
        return [float(canon[(BEAM_SHIFT - j) % n]) for j in range(n)]
    def getImage(self):
        bgr = self.r.cam.render(self.r.sim.true_pose())
        bgra = np.concatenate([bgr, np.full(bgr.shape[:2] + (1,), 255, np.uint8)], axis=2)
        return bgra.tobytes()


class Robot:
    def __init__(self):
        self.world = World()
        rng = np.random.default_rng(0)
        self.sim = RobotSim((*C.START, math.pi / 2), self.world, rng, w_bias=0.02, eps=0.01)
        self.lidar, self.cam = SimLidar(self.world, rng), SimCamera(self.world, rng)
        self.m = {"left wheel motor": _Motor(), "right wheel motor": _Motor()}

    def getBasicTimeStep(self): return 32.0
    def getTime(self): return self.sim.t
    def getDevice(self, name):
        return self.m[name] if name in self.m else _Dev(self, name)
    def step(self, ms):
        wl, wr = self.m["left wheel motor"].v, self.m["right wheel motor"].v
        v = C.WHEEL_R * (wl + wr) / 2
        w = C.WHEEL_R * (wr - wl) / C.TRACK
        self.sim.step(v, w)
        return -1 if self.sim.t > T_MAX else 0
