"""Webots controller: perception (LiDAR + camera fusion) -> A* planning -> PID path tracking.

Runs the SAME navlib code that produced the offline results.

Phases
  0 CALIBRATE  registers the LiDAR beam order against the known start pose and measures the IMU yaw sign
               (conventions differ between Webots versions, so they are measured rather than assumed)
  1 SURVEY     drives the pre-defined survey route, spinning 360 deg at scan stops, fusing LiDAR + camera
  2 PLAN       A* on the fused + inflated grid, line-of-sight smoothing
  3 EXECUTE    look-ahead PID tracking of the planned path
Outputs (written to <project>/logs/):  webots_trajectory.csv, webots_plan.csv, webots_grid.npz
(file prefix 'webots' can be overridden with the NAV_LOG_PREFIX environment variable)
"""
import os
import sys
import json
import math
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from controller import Robot  # noqa: E402  (provided by Webots)

from navlib import config as C  # noqa: E402
from navlib.control import PathTracker, TurnInPlace, wheel_speeds, wrap  # noqa: E402
from navlib.perception import OccupancyGrid, preprocess_lidar, detect  # noqa: E402
from navlib.planning import path_length, resample  # noqa: E402
from navlib.sensors import LIDAR_ANGLES  # noqa: E402
from navlib.sim import plan_on_grid, cross_track  # noqa: E402
from navlib.world import FULL_SURVEY, World  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
PREFIX = os.environ.get("NAV_LOG_PREFIX", "webots")
os.makedirs(LOG_DIR, exist_ok=True)


def load_gains():
    try:
        g = json.load(open(os.path.join(ROOT, "navlib", "tuned_gains.json")))
        return (g["Kp"], g["Ki"], g["Kd"])
    except Exception:
        return (5.0, 0.1, 0.4)


class NavRobot:
    def __init__(self):
        self.robot = Robot()
        self.ts = int(self.robot.getBasicTimeStep())
        assert abs(self.ts / 1000.0 - C.DT) < 1e-6, "world basicTimeStep must be 32 ms (navlib.config.DT)"
        self.lm = self.robot.getDevice("left wheel motor")
        self.rm = self.robot.getDevice("right wheel motor")
        for m in (self.lm, self.rm):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.gps = self.robot.getDevice("gps"); self.gps.enable(self.ts)
        self.imu = self.robot.getDevice("imu"); self.imu.enable(self.ts)
        self.lidar = self.robot.getDevice("lidar"); self.lidar.enable(self.ts)
        self.cam = self.robot.getDevice("camera"); self.cam.enable(self.ts)
        self.gains = load_gains()
        self.grid = OccupancyGrid()
        self.yaw_sign, self.yaw_off, self.yaw0 = 1.0, 0.0, 0.0
        self.beam_sign, self.beam_shift = 1, 0
        self.k = 0
        self.traj = []
        self.alive = True
        for _ in range(3):                      # let sensors produce their first valid readings
            self.robot.step(self.ts)

    # ------------------------------------------------------------------ low level
    def time(self):
        return self.robot.getTime()

    def pose(self):
        x, y, _ = self.gps.getValues()
        yaw = self.imu.getRollPitchYaw()[2]
        th = wrap(math.pi / 2 + self.yaw_sign * (yaw - self.yaw0)) if self._calibrated_yaw else yaw
        return (x, y, th)

    _calibrated_yaw = False

    def drive(self, v, w):
        wl, wr = wheel_speeds(v, w)
        self.lm.setVelocity(wl)
        self.rm.setVelocity(wr)

    def tick(self, v, w, sense=False):
        """One control step. Returns False when the simulation is over."""
        self.drive(v, w)
        if self.robot.step(self.ts) == -1:
            self.alive = False
            return False
        self.k += 1
        x, y, th = self.pose()
        self.traj.append((self.time(), x, y, th))
        if sense and self.k % 6 == 0:
            self.sense()
        return True

    # ------------------------------------------------------------------ sensing
    def lidar_canonical(self):
        raw = np.array(self.lidar.getRangeImage(), float)
        raw[~np.isfinite(raw)] = np.inf
        idx = (self.beam_sign * np.arange(C.LIDAR_BEAMS) + self.beam_shift) % C.LIDAR_BEAMS
        out = raw[idx]
        out[(out < C.LIDAR_RMIN) | (out > C.LIDAR_RMAX)] = np.inf
        return out

    def camera_bgr(self):
        buf = self.cam.getImage()                                    # BGRA bytes
        img = np.frombuffer(buf, np.uint8).reshape(C.CAM_H, C.CAM_W, 4)
        return np.ascontiguousarray(img[:, :, :3])

    def sense(self):
        pose = self.pose()
        self.grid.update_lidar(pose, preprocess_lidar(self.lidar_canonical()))
        self.grid.update_camera(pose, detect(self.camera_bgr()))

    # ------------------------------------------------------------------ phase 0
    def calibrate(self):
        """(a) find the LiDAR beam permutation (direction + start index) by matching the scan taken at the
        known start pose against the scan the ground-truth world model predicts there;
        (b) spin briefly to learn the sign of the IMU yaw."""
        world = World()
        x, y, th0 = C.START[0], C.START[1], math.pi / 2
        exp, _ = world.raycast((x, y), th0 + LIDAR_ANGLES, min_height=C.LIDAR_HEIGHT, rmax=C.LIDAR_RMAX)
        exp = np.where(np.isfinite(exp), exp, C.LIDAR_RMAX)
        raw = np.array(self.lidar.getRangeImage(), float)
        raw = np.where(np.isfinite(raw), raw, C.LIDAR_RMAX)
        n = C.LIDAR_BEAMS
        best = (1e9, 1, 0)
        for sgn in (1, -1):
            for sh in range(n):
                cand = raw[(sgn * np.arange(n) + sh) % n]
                err = float(np.mean(np.abs(np.minimum(cand, C.LIDAR_RMAX) - exp)))
                if err < best[0]:
                    best = (err, sgn, sh)
        self.beam_sign, self.beam_shift = best[1], best[2]
        print(f"[calibrate] LiDAR beam mapping: sign={best[1]} shift={best[2]} mean |error|={best[0]:.3f} m")
        if best[0] > 0.15:
            print("[calibrate] WARNING: large LiDAR mismatch - is the robot at the start pose in arena.wbt?")

        yaw_a = self.imu.getRollPitchYaw()[2]
        for _ in range(16):                                           # ~0.5 s of +1 rad/s (CCW)
            self.tick(0.0, 1.0)
        yaw_b = self.imu.getRollPitchYaw()[2]
        d = wrap(yaw_b - yaw_a)
        self.yaw_sign = 1.0 if d > 0 else -1.0
        self.yaw0 = yaw_a                                              # reference = reading BEFORE the spin
        self._calibrated_yaw = True
        # heading before the spin was pi/2 by construction of arena.wbt; the spin is then part of the pose
        print(f"[calibrate] IMU yaw sign = {self.yaw_sign:+.0f} (d_yaw={d:+.3f} rad for a +CCW command)")
        for _ in range(10):
            self.tick(0.0, 0.0)
        self.traj = []                                                 # drop the calibration spin from the log

    # ------------------------------------------------------------------ phases 1-3
    def turn_to(self, ref, sense=True, timeout=20.0):
        t = TurnInPlace(self.gains, ref)
        for _ in range(int(timeout / C.DT)):
            v, w, done, _e = t.command(self.pose(), C.DT)
            if not self.tick(v, w, sense) or done:
                return

    def follow(self, path, v_max, sense=True, timeout=300.0, log=None):
        trk = PathTracker(path, self.gains, v_max=v_max, lookahead=0.4)
        for _ in range(int(timeout / C.DT)):
            pose = self.pose()
            v, w, done, e = trk.command(pose, C.DT)
            if log is not None:
                log.append((self.time(), pose[0], pose[1], pose[2], v, w, e, cross_track(path, pose[0], pose[1])))
            if not self.tick(v, w, sense) or done:
                return done
        return False

    def survey(self):
        print("[survey] building the fused occupancy map ...")
        for i in range(1, len(FULL_SURVEY)):
            (x0, y0, _), (x1, y1, scan) = FULL_SURVEY[i - 1], FULL_SURVEY[i]
            if not self.alive:
                return
            self.turn_to(math.atan2(y1 - y0, x1 - x0))
            # start the leg from where the robot actually is, aimed at the next waypoint
            px, py, _ = self.pose()
            leg = resample(np.array([[px, py], [x1, y1]]), 0.05)
            self.follow(leg, v_max=0.4)
            if scan:
                th0 = self.pose()[2]
                for j in range(1, 9):
                    self.turn_to(wrap(th0 + j * math.pi / 4))
                    for _ in range(12):
                        self.tick(0.0, 0.0, sense=True)
        print(f"[survey] done at t={self.time():.1f} s; landmarks: "
              f"{ {k: np.round(v, 2).tolist() for k, v in self.grid.landmark_positions().items()} }")

    def run(self):
        self.calibrate()
        self.survey()
        if not self.alive:
            return
        blocked = self.grid.planning_grid("fused")
        # plan from where the robot actually is
        px, py, _ = self.pose()
        res, raw, smooth = plan_on_grid(blocked, (px, py), C.GOAL, "astar")
        if smooth is None:
            print("[plan] NO PATH FOUND - check the map in logs/webots_grid.npz")
            return
        print(f"[plan] A*: {res['expanded']} nodes expanded, {res['time'] * 1000:.1f} ms, "
              f"grid cost {res['cost']:.2f} m, smoothed length {path_length(smooth):.2f} m")
        np.savetxt(os.path.join(LOG_DIR, f"{PREFIX}_plan.csv"), smooth, delimiter=",", header="x,y", comments="")
        np.savez(os.path.join(LOG_DIR, f"{PREFIX}_grid.npz"), L_lidar=self.grid.L_lidar, L_cam=self.grid.L_cam,
                 blocked=blocked, raw=raw, smooth=smooth, survey_traj=np.array(self.traj))
        dense = resample(smooth, 0.05)
        log = []
        self.turn_to(math.atan2(dense[min(8, len(dense) - 1)][1] - py, dense[min(8, len(dense) - 1)][0] - px), sense=False)
        t_start = self.time()
        ok = self.follow(dense, v_max=C.V_MAX, sense=False, log=log)
        print(f"[execute] reached goal={ok}  time-to-goal={self.time() - t_start:.1f} s")
        with open(os.path.join(LOG_DIR, f"{PREFIX}_trajectory.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "x", "y", "theta", "v_cmd", "w_cmd", "heading_err", "cross_track"])
            wr.writerows(log)
        for _ in range(10):
            self.tick(0.0, 0.0)
        print("[done] logs written to", LOG_DIR)


if __name__ == "__main__":
    NavRobot().run()
