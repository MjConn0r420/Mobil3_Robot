"""Synthetic LiDAR and camera used by the offline simulator.

In Webots the real `Lidar` / `Camera` devices replace these classes; the output
format is identical (range array / BGR image) so perception.py is unchanged.
"""
import numpy as np
from . import config as C

LIDAR_ANGLES = np.linspace(-np.pi, np.pi, C.LIDAR_BEAMS, endpoint=False)   # robot frame, CCW+

BGR = {"orange": (0, 128, 255), "blue": (255, 0, 0), "green": (0, 200, 0),
       "magenta": (255, 0, 255), "gray": (95, 95, 95)}
FLOOR_BGR, BG_BGR = (200, 200, 200), (225, 225, 225)


class SimLidar:
    def __init__(self, world, rng, sigma=C.LIDAR_SIGMA):
        self.world, self.rng, self.sigma = world, rng, sigma

    def scan(self, pose):
        x, y, th = pose
        r, _ = self.world.raycast((x, y), th + LIDAR_ANGLES, min_height=C.LIDAR_HEIGHT,
                                  rmax=C.LIDAR_RMAX)
        fin = np.isfinite(r)
        r = r.copy()
        r[fin] += self.rng.normal(0, self.sigma, fin.sum())
        r[self.rng.random(len(r)) < C.LIDAR_DROPOUT] = np.inf               # dropouts
        spike = self.rng.random(len(r)) < C.LIDAR_SPIKE                      # random outliers
        r[spike] = self.rng.uniform(0.3, C.LIDAR_RMAX, spike.sum())
        r[(r < C.LIDAR_RMIN) | (r > C.LIDAR_RMAX)] = np.inf
        return r


class SimCamera:
    """Pinhole camera with horizontal optical axis; renders a BGR image from the
    exact scene geometry (perspective-correct vertical extent of every object)."""

    def __init__(self, world, rng, pix_sigma=C.CAM_PIX_SIGMA):
        self.world, self.rng, self.pix_sigma = world, rng, pix_sigma
        u = np.arange(C.CAM_W) + 0.5
        self.alpha = -np.arctan((u - C.CAM_W / 2) / C.CAM_F)                # right = negative bearing
        self.cosa = np.cos(self.alpha)

    def render(self, pose):
        x, y, th = pose
        W, H, f, h = C.CAM_W, C.CAM_H, C.CAM_F, C.CAM_HEIGHT
        cy = H / 2.0
        org = (x + C.CAM_XOFF * np.cos(th), y + C.CAM_XOFF * np.sin(th))
        r, oid = self.world.raycast(org, th + self.alpha, min_height=0.0)
        img = np.empty((H, W, 3), np.uint8)
        img[: H // 2] = BG_BGR
        img[H // 2:] = FLOOR_BGR
        for u in range(W):
            if oid[u] < 0:
                continue
            ob = self.world.objects[oid[u]]
            z = r[u] * self.cosa[u]                                           # depth along optical axis
            top = int(round(cy - f * (ob["h"] - h) / z))
            bot = int(round(cy + f * h / z))
            top, bot = max(top, 0), min(bot, H)
            if bot <= top:
                continue
            shade = max(0.55, 1.0 - 0.05 * z)
            col = np.array(BGR[ob["color"]], float)
            if ob["color"] != "gray":                                         # keep hue/saturation, dim value
                col = col * shade
            img[top:bot, u] = col.astype(np.uint8)
        noisy = img.astype(np.float32) + self.rng.normal(0, self.pix_sigma, img.shape)
        salt = self.rng.random((H, W))
        noisy[salt < C.CAM_SALT / 2] = 0
        noisy[salt > 1 - C.CAM_SALT / 2] = 255
        return np.clip(noisy, 0, 255).astype(np.uint8)
