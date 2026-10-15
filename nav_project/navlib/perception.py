"""Perception: LiDAR pre-processing, OpenCV colour detection, ground-plane
back-projection, and log-odds occupancy-grid sensor fusion."""
import math
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.ndimage import median_filter
from . import config as C
from .sensors import LIDAR_ANGLES
from .world import LANDMARK_RADIUS

# ---------------------------------------------------------------- LiDAR
def preprocess_lidar(ranges):
    """Spike rejection + validity gating.
    * dropouts (inf) are mapped to RMAX+1 so the median filter sees them as 'far'
    * a 3-tap circular median removes isolated 1-beam outliers but keeps edges
      (a 0.4 m landmark at 4 m still spans ~6 beams, so nothing real is lost)."""
    r = np.where(np.isfinite(ranges), ranges, C.LIDAR_RMAX + 1.0)
    r = median_filter(r, size=3, mode="wrap")
    valid = (r >= C.LIDAR_RMIN) & (r < C.LIDAR_RMAX - 0.02)
    return np.where(valid, r, np.inf)


# ---------------------------------------------------------------- camera
HSV_RANGES = {   # OpenCV hue is 0..179
    "obstacle": ((5, 120, 70), (22, 255, 255)),
    "blue": ((100, 120, 60), (130, 255, 255)),
    "green": ((45, 120, 60), (75, 255, 255)),
    "magenta": ((140, 120, 60), (165, 255, 255)),
}
CAM_H_BELIEF = C.CAM_HEIGHT     # camera height the perception code ASSUMES (sim can set it wrong to model mis-calibration)
MIN_BLOB_AREA = 40
K_OPEN = np.ones((3, 3), np.uint8)
K_CLOSE = np.ones((5, 5), np.uint8)


@dataclass
class Detection:
    cls: str
    cols: np.ndarray       # pixel column of each contact point
    rows: np.ndarray       # image row of the obstacle/floor boundary at that column
    bbox: tuple
    area: int


def camera_masks(bgr):
    """Blur -> HSV -> per-class threshold -> open/close. Returns dict cls->mask and the blurred image."""
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)          # suppress Gaussian pixel noise before thresholding
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)      # hue is far more illumination-robust than RGB
    masks = {}
    for cls, (lo, hi) in HSV_RANGES.items():
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K_OPEN)    # remove salt noise
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K_CLOSE)  # bridge pepper holes inside blobs
        masks[cls] = m
    return masks, blur


def detect(bgr):
    masks, _ = camera_masks(bgr)
    dets = []
    for cls, m in masks.items():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            if a < MIN_BLOB_AREA:
                continue
            sub = lab[y:y + h, x:x + w] == k
            has = sub.any(0)
            cols = np.nonzero(has)[0]
            bottom = sub.shape[0] - 1 - np.argmax(sub[::-1, :], axis=0)[cols]   # last True row per column
            rows = y + bottom + 1                                               # boundary = row edge below
            dets.append(Detection(cls, x + cols, rows, (x, y, w, h), int(a)))
    return dets


def contact_points(det):
    """Back-project the blob/floor boundary to the ground plane (robot frame).
    Pinhole + known camera height h:  z = f*h/(v - cy),  lateral = (u - cx)/f * z.
    Boundary pixels touching the image bottom are clipped (object too close) and skipped,
    as are points beyond CAM_ZMAX where one pixel of row error exceeds ~5 % range error."""
    f, h = C.CAM_F, CAM_H_BELIEF
    cx, cy = C.CAM_W / 2.0, C.CAM_H / 2.0
    ok = (det.rows < C.CAM_H) & (det.rows - cy > 1.0)
    if not ok.any():
        return np.zeros((0, 2))
    u = det.cols[ok] + 0.5
    v = det.rows[ok].astype(float)
    z = f * h / (v - cy)
    keep = z <= C.CAM_ZMAX
    z, u = z[keep], u[keep]
    X = z + C.CAM_XOFF
    Y = -(u - cx) / f * z
    return np.stack([X, Y], 1)


def to_world(pts_robot, pose):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return np.stack([x + c * pts_robot[:, 0] - s * pts_robot[:, 1],
                     y + s * pts_robot[:, 0] + c * pts_robot[:, 1]], 1)


def landmark_estimate(pts_robot, pose):
    """Landmark centre = mean of visible contact points pushed outwards by the known radius."""
    if len(pts_robot) < 3:
        return None
    m = pts_robot.mean(0)
    d = np.array([m[0] - C.CAM_XOFF, m[1]])
    d /= np.linalg.norm(d) + 1e-9
    c_robot = m + LANDMARK_RADIUS * d
    return to_world(c_robot[None], pose)[0]


# ---------------------------------------------------------------- occupancy grid fusion
class OccupancyGrid:
    """Two log-odds layers (one per sensor) fused by a conservative OR (max).

    Why OR/max and not a sum?  The LiDAR can only prove a cell free *at its scan-plane
    height*; an object lower than that plane gets repeatedly 'cleared' by LiDAR rays.
    Summing would let that free-evidence cancel the camera's occupied-evidence and the
    robot would drive over the object. For collision avoidance a false negative is far
    more costly than a false positive, so a cell is blocked if EITHER sensor believes it is."""

    def __init__(self):
        self.n = C.N
        self.L_lidar = np.zeros((self.n, self.n), np.float32)
        self.L_cam = np.zeros((self.n, self.n), np.float32)
        self.landmarks = {}            # cls -> list of world estimates

    # --- helpers
    def _flat(self, pts):
        c = np.clip(np.floor(pts[:, 0] / C.RES).astype(int), 0, self.n - 1)
        r = np.clip(np.floor(pts[:, 1] / C.RES).astype(int), 0, self.n - 1)
        return r * self.n + c

    def _ray_update(self, L, origin, ang, rng, hit, l_occ, l_free):
        """Free-space along each ray (sampled every RES/2, de-duplicated per update)
        and occupied evidence at the hit cell."""
        step = C.RES / 2
        smax = np.nanmax(np.where(np.isfinite(rng), rng, 0))
        if smax <= 0:
            return
        s = np.arange(0.0, smax + step, step)
        reach = np.where(hit, rng - 0.6 * C.RES, rng)          # stop just before the hit cell
        mask = s[None, :] <= reach[:, None]
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        pts = origin[None, None, :] + s[None, :, None] * d[:, None, :]
        free = np.unique(self._flat(pts[mask]))
        occ = np.unique(self._flat((origin[None] + rng[hit, None] * d[hit])) ) if hit.any() else np.array([], int)
        free = np.setdiff1d(free, occ, assume_unique=True)
        flat = L.reshape(-1)
        flat[free] = np.clip(flat[free] + l_free, C.L_MIN, C.L_MAX)
        flat[occ] = np.clip(flat[occ] + l_occ, C.L_MIN, C.L_MAX)

    # --- sensor updates
    def update_lidar(self, pose, ranges_clean):
        x, y, th = pose
        hit = np.isfinite(ranges_clean)
        rng = np.where(hit, ranges_clean, C.LIDAR_RMAX)
        self._ray_update(self.L_lidar, np.array([x, y]), th + LIDAR_ANGLES, rng, hit,
                         C.L_OCC_LIDAR, C.L_FREE_LIDAR)

    def update_camera(self, pose, dets):
        """dets: list of Detection. Every detected class is an obstacle; landmark classes
        additionally feed the landmark list."""
        x, y, th = pose
        org = np.array([x + C.CAM_XOFF * math.cos(th), y + C.CAM_XOFF * math.sin(th)])
        all_w = []
        for det in dets:
            pr = contact_points(det)
            if len(pr) == 0:
                continue
            pw = to_world(pr, pose)
            all_w.append(pw)
            if det.cls != "obstacle":
                est = landmark_estimate(pr, pose)
                if est is not None:
                    self.landmarks.setdefault(det.cls, []).append(est)
        if not all_w:
            return
        pw = np.vstack(all_w)
        v = pw - org
        rng = np.linalg.norm(v, axis=1)
        ang = np.arctan2(v[:, 1], v[:, 0])
        self._ray_update(self.L_cam, org, ang, rng, np.ones(len(rng), bool),
                         C.L_OCC_CAM, C.L_FREE_CAM)

    # --- products
    def occupied(self, mode="fused"):
        if mode == "lidar":
            return self.L_lidar >= C.L_THRESH
        if mode == "camera":
            return self.L_cam >= C.L_THRESH
        return (np.maximum(self.L_lidar, self.L_cam) >= C.L_THRESH)

    def known(self):
        return (self.L_lidar != 0) | (self.L_cam != 0)

    def planning_grid(self, mode="fused", inflate=C.INFLATE):
        occ = self.occupied(mode).astype(np.uint8)
        k = int(math.ceil(inflate / C.RES))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = ((xx ** 2 + yy ** 2) * C.RES ** 2 <= inflate ** 2 + 1e-9).astype(np.uint8)
        return cv2.dilate(occ, disk).astype(bool)

    def landmark_positions(self):
        return {c: np.median(np.array(v), axis=0) for c, v in self.landmarks.items() if len(v) >= 3}
