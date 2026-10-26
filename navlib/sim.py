"""Offline closed-loop simulator (kinematics + motor lag + delay + noise) and the
high-level routines (survey -> map -> plan -> track) shared with the Webots controller logic."""
import math
from collections import deque
import numpy as np
from . import config as C
from .control import PathTracker, TurnInPlace, wheel_speeds, wrap
from .sensors import SimLidar, SimCamera
from .perception import preprocess_lidar, detect, OccupancyGrid
from .planning import astar, dijkstra, cells_to_xy, smooth_path, resample, path_length
from .world import FULL_SURVEY, cell_of

TUNED_GAINS = (3.0, 0.3, 0.3)       # overwritten by the tuning experiment (see run_all.py)


class RobotSim:
    def __init__(self, pose, world=None, rng=None, w_bias=0.0, eps=0.0,
                 pose_noise=(C.POSE_SIGMA_XY, C.POSE_SIGMA_TH)):
        self.x, self.y, self.th = pose
        self.world, self.rng = world, rng or np.random.default_rng(0)
        self.wl = self.wr = 0.0
        self.q = deque([(0.0, 0.0)] * C.DELAY_STEPS)
        self.w_bias, self.eps, self.pn = w_bias, eps, pose_noise
        self.t = 0.0
        self.v = self.w = 0.0
        self.min_clear = math.inf
        self.collided = False

    def step(self, v_cmd, w_cmd, dt=C.DT):
        self.q.append(wheel_speeds(v_cmd, w_cmd))
        wl_c, wr_c = self.q.popleft()
        a = dt / (C.TAU_MOTOR + dt)
        self.wl += (wl_c - self.wl) * a
        self.wr += (wr_c - self.wr) * a
        self.v = C.WHEEL_R * (self.wl + (1 + self.eps) * self.wr) / 2
        self.w = C.WHEEL_R * ((1 + self.eps) * self.wr - self.wl) / C.TRACK + self.w_bias
        thm = self.th + 0.5 * self.w * dt
        self.x += self.v * math.cos(thm) * dt
        self.y += self.v * math.sin(thm) * dt
        self.th = wrap(self.th + self.w * dt)
        self.t += dt
        if self.world is not None:
            c = self.world.clearance(self.x, self.y)
            self.min_clear = min(self.min_clear, c)
            if c < C.R_COLLIDE:
                self.collided = True

    def true_pose(self):
        return (self.x, self.y, self.th)

    def measure(self):
        sx, st = self.pn
        return (self.x + self.rng.normal(0, sx), self.y + self.rng.normal(0, sx),
                wrap(self.th + self.rng.normal(0, st)))


# ------------------------------------------------------------------ survey / mapping
def run_survey(world, seed=0, gains=TUNED_GAINS, lidar_sigma=C.LIDAR_SIGMA,
               pix_sigma=C.CAM_PIX_SIGMA, v_max=0.4, sense_every=6, log_detections=None):
    """Drive the pre-defined survey route out and back (spinning 360 deg at 'scan' stops on the way out) while fusing
    LiDAR and camera into the occupancy grid; the robot ends at the start pose. Returns (grid, sim, trajectory)."""
    rng = np.random.default_rng(seed)
    sim = RobotSim((*C.START, math.pi / 2), world, rng)
    lidar, cam = SimLidar(world, rng, lidar_sigma), SimCamera(world, rng, pix_sigma)
    grid = OccupancyGrid()
    traj = []
    k = [0]

    def sense():
        pose = sim.measure()
        grid.update_lidar(pose, preprocess_lidar(lidar.scan(sim.true_pose())))
        dets = detect(cam.render(sim.true_pose()))
        grid.update_camera(pose, dets)
        if log_detections is not None:
            log_detections.append((sim.true_pose(), dets))

    def tick(v, w):
        sim.step(v, w)
        traj.append((sim.t, sim.x, sim.y, sim.th))
        k[0] += 1
        if k[0] % sense_every == 0:
            sense()

    for i in range(1, len(FULL_SURVEY)):
        (x0, y0, _), (x1, y1, scan) = FULL_SURVEY[i - 1], FULL_SURVEY[i]
        # turn to face the leg, then drive it
        ref = math.atan2(y1 - y0, x1 - x0)
        turn = TurnInPlace(gains, ref)
        for _ in range(int(20 / C.DT)):
            v, w, done, _e = turn.command(sim.measure(), C.DT)
            tick(v, w)
            if done:
                break
        leg = resample(np.array([[x0, y0], [x1, y1]]), 0.05)
        trk = PathTracker(leg, gains, v_max=v_max, lookahead=0.4)
        for _ in range(int(60 / C.DT)):
            v, w, done, _e = trk.command(sim.measure(), C.DT)
            tick(v, w)
            if done:
                break
        if scan:
            th0 = sim.th
            for j in range(1, 9):                      # 8 x 45 deg  -> full 360 deg with camera overlap
                turn = TurnInPlace(gains, wrap(th0 + j * math.pi / 4))
                for _ in range(int(10 / C.DT)):
                    v, w, done, _e = turn.command(sim.measure(), C.DT)
                    tick(v, w)
                    if done:
                        break
                for _ in range(2):                     # two sensing cycles per stop
                    for _ in range(sense_every):
                        sim.step(0, 0)
                    sense()
    return grid, sim, np.array(traj)


# ------------------------------------------------------------------ planning + tracking
def plan_on_grid(blocked, start_xy, goal_xy, algo="astar", heur="octile"):
    s, g = cell_of(*start_xy), cell_of(*goal_xy)
    res = astar(blocked, s, g, heur) if algo == "astar" else dijkstra(blocked, s, g)
    if not res["found"]:
        return res, None, None
    raw = cells_to_xy(res["path"])
    raw[0], raw[-1] = start_xy, goal_xy
    smooth = smooth_path(raw, blocked)
    return res, raw, smooth


def track(world, path_xy, gains=TUNED_GAINS, seed=1, v_max=C.V_MAX, w_bias=0.03, eps=0.01,
          t_max=240.0, start_theta=None, anti_windup=True, lookahead=0.4):
    """Closed-loop tracking of `path_xy`. Stops on collision, goal or timeout."""
    rng = np.random.default_rng(seed)
    p0 = path_xy[0]
    th0 = start_theta if start_theta is not None else math.atan2(path_xy[1][1] - p0[1], path_xy[1][0] - p0[0])
    sim = RobotSim((p0[0], p0[1], th0), world, rng, w_bias=w_bias, eps=eps)
    dense = resample(path_xy, 0.05)
    trk = PathTracker(dense, gains, v_max=v_max, lookahead=lookahead, anti_windup=anti_windup)
    log = []
    reached = False
    for _ in range(int(t_max / C.DT)):
        v, w, done, e = trk.command(sim.measure(), C.DT)
        sim.step(v, w)
        cte = cross_track(dense, sim.x, sim.y)
        log.append((sim.t, sim.x, sim.y, sim.th, sim.v, sim.w, e, cte))
        if sim.collided:
            break
        if done:
            reached = True
            break
    return dict(log=np.array(log), reached=reached, collided=sim.collided,
                min_clear=sim.min_clear, time=sim.t)


def cross_track(poly, x, y):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / (ab ** 2).sum(1).clip(1e-12), 0, 1)
    px, py = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    return float(np.sqrt((px - x) ** 2 + (py - y) ** 2).min())


def tracking_stats(res, ref_path):
    L = res["log"]
    cte = L[:, 7]
    actual = float(np.hypot(*np.diff(L[:, 1:3], axis=0).T).sum())
    return dict(reached=res["reached"], collided=res["collided"], time_s=round(res["time"], 1),
                planned_len=round(path_length(ref_path), 2), actual_len=round(actual, 2),
                cte_mean=round(float(cte.mean()), 3), cte_rms=round(float(np.sqrt((cte ** 2).mean())), 3),
                cte_max=round(float(cte.max()), 3), min_clear=round(res["min_clear"], 3))
