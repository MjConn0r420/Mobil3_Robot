"""Ground-truth world description + exact 2-D ray casting.

The world is described ONCE here; `worlds/generate_world.py` writes the
identical layout to a Webots .wbt file, and the offline simulator uses the
same description for its synthetic LiDAR / camera.

Objects
  box : axis-aligned rectangle  (cx, cy, sx, sy)
  cyl : vertical cylinder       (cx, cy, r)
Every object has a height `h` and a colour class
  'orange'  painted obstacle (camera obstacle class)
  'blue' / 'green' / 'magenta'  landmarks (cylinders)
  'gray'    unpainted arena boundary wall
The *LOW* box has h = 0.12 m < LiDAR scan height (0.20 m): the LiDAR plane
passes straight over it, only the camera can see it.  This is what makes
sensor fusion necessary rather than decorative.
"""
import numpy as np
from . import config as C

WALL_T = 0.10
WALL_H = 0.50


def _box(name, cx, cy, sx, sy, h, color):
    return dict(name=name, kind="box", cx=cx, cy=cy, sx=sx, sy=sy, h=h, color=color)


def _cyl(name, cx, cy, r, h, color):
    return dict(name=name, kind="cyl", cx=cx, cy=cy, r=r, h=h, color=color)


def default_objects():
    A = C.ARENA
    objs = [
        # --- arena boundary (outside the [0,A]^2 floor) ---
        _box("wall_S", A / 2, -WALL_T / 2, A + 2 * WALL_T, WALL_T, WALL_H, "gray"),
        _box("wall_N", A / 2, A + WALL_T / 2, A + 2 * WALL_T, WALL_T, WALL_H, "gray"),
        _box("wall_W", -WALL_T / 2, A / 2, WALL_T, A, WALL_H, "gray"),
        _box("wall_E", A + WALL_T / 2, A / 2, WALL_T, A, WALL_H, "gray"),
        # --- serpentine dividers (gap on the right at y=5, gap on the left at y=7.5) ---
        _box("divider_1", 3.25, 5.0, 6.5, 0.2, WALL_H, "orange"),
        _box("divider_2", 6.75, 7.5, 6.5, 0.2, WALL_H, "orange"),
        # --- LOW box in the left gap: invisible to the LiDAR plane ---
        _box("low_box", 3.0, 7.5, 1.4, 0.5, 0.12, "orange"),
        # --- clutter ---
        _box("box_A", 3.2, 2.4, 1.2, 0.8, WALL_H, "orange"),
        _box("box_C", 7.4, 1.4, 0.8, 0.8, WALL_H, "orange"),
        _box("box_E", 6.0, 8.5, 1.0, 0.8, WALL_H, "orange"),
        _box("box_F", 1.8, 3.3, 0.6, 0.6, WALL_H, "orange"),
        # --- landmarks (known radius 0.20 m) ---
        _cyl("lm_blue", 5.5, 2.0, 0.20, 0.60, "blue"),
        _cyl("lm_green", 8.2, 3.4, 0.20, 0.60, "green"),
        _cyl("lm_magenta", 5.0, 6.4, 0.20, 0.60, "magenta"),
    ]
    return objs


# Survey route used while building the map: (x, y, scan_here)
SURVEY_ROUTE = [
    (1.0, 1.0, False), (1.0, 4.0, True), (4.5, 4.0, False), (9.0, 4.0, True),
    (9.0, 5.65, False), (6.0, 5.65, True), (1.0, 5.65, True), (1.0, 8.8, True),
    (1.0, 9.4, False), (5.0, 9.4, False), (9.0, 9.4, True),
]

# The robot drives the route out and then back along the same (verified collision-free) legs, so that
# planning starts from the real start pose. Scans are only taken on the way out.
FULL_SURVEY = SURVEY_ROUTE + [(x, y, False) for (x, y, _s) in reversed(SURVEY_ROUTE[:-1])]

LANDMARK_RADIUS = 0.20


class World:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else default_objects()
        P, E, SO, SH = [], [], [], []
        cc, cr, co, ch = [], [], [], []
        for i, o in enumerate(self.objects):
            if o["kind"] == "box":
                x0, x1 = o["cx"] - o["sx"] / 2, o["cx"] + o["sx"] / 2
                y0, y1 = o["cy"] - o["sy"] / 2, o["cy"] + o["sy"] / 2
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                for k in range(4):
                    p, q = np.array(pts[k]), np.array(pts[(k + 1) % 4])
                    P.append(p); E.append(q - p); SO.append(i); SH.append(o["h"])
            else:
                cc.append((o["cx"], o["cy"])); cr.append(o["r"]); co.append(i); ch.append(o["h"])
        self.seg_p, self.seg_e = np.array(P), np.array(E)
        self.seg_obj, self.seg_h = np.array(SO), np.array(SH)
        self.circ_c = np.array(cc).reshape(-1, 2)
        self.circ_r, self.circ_obj, self.circ_h = np.array(cr), np.array(co), np.array(ch)

    # ------------------------------------------------------------------
    def raycast(self, origin, angles, min_height=0.0, rmax=np.inf):
        """Nearest hit for rays from `origin` at world `angles`.
        Only objects with height >= min_height are hit.
        Returns (dist[N], obj_index[N]); dist=inf, obj=-1 when nothing is hit."""
        angles = np.asarray(angles, float)
        d = np.stack([np.cos(angles), np.sin(angles)], 1)          # (N,2)
        o = np.asarray(origin, float)
        best = np.full(len(angles), np.inf)
        obj = np.full(len(angles), -1, int)

        m = self.seg_h >= min_height
        if m.any():
            p, e, so = self.seg_p[m], self.seg_e[m], self.seg_obj[m]
            w = p - o                                              # (S,2)
            denom = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / denom
                s = (w[None, :, 0] * d[:, 1:2] - w[None, :, 1] * d[:, 0:1]) / denom
            ok = (np.abs(denom) > 1e-12) & (t > 1e-6) & (s >= 0) & (s <= 1)
            t = np.where(ok, t, np.inf)
            k = t.argmin(1)
            tb = t[np.arange(len(angles)), k]
            upd = tb < best
            best[upd] = tb[upd]; obj[upd] = so[k][upd]

        mc = self.circ_h >= min_height
        if mc.any():
            c, r, co = self.circ_c[mc], self.circ_r[mc], self.circ_obj[mc]
            oc = o - c                                             # (C,2)
            b = d[:, 0:1] * oc[None, :, 0] + d[:, 1:2] * oc[None, :, 1]
            c0 = (oc ** 2).sum(1) - r ** 2
            disc = b ** 2 - c0[None, :]
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(disc)
            ok = (disc >= 0) & (t > 1e-6)
            t = np.where(ok, t, np.inf)
            k = t.argmin(1)
            tb = t[np.arange(len(angles)), k]
            upd = tb < best
            best[upd] = tb[upd]; obj[upd] = co[k][upd]

        far = best > rmax
        best[far] = np.inf; obj[far] = -1
        return best, obj

    # ------------------------------------------------------------------
    def clearance(self, x, y):
        """Distance from point (x,y) to the nearest obstacle surface (any height)."""
        best = min(x, y, C.ARENA - x, C.ARENA - y)
        for o in self.objects:
            if o["name"].startswith("wall"):
                continue
            if o["kind"] == "box":
                dx = max(abs(x - o["cx"]) - o["sx"] / 2, 0.0)
                dy = max(abs(y - o["cy"]) - o["sy"] / 2, 0.0)
                dist = float(np.hypot(dx, dy))
                if dx == 0 and dy == 0:
                    dist = 0.0
            else:
                dist = float(np.hypot(x - o["cx"], y - o["cy"]) - o["r"])
            best = min(best, dist)
        return best

    def gt_occupancy(self, res=C.RES, ss=5):
        """Ground-truth occupancy raster (True = some part of the cell is covered
        by an obstacle of ANY height). The boundary ring of cells is occupied."""
        n = int(round(C.ARENA / res))
        sub = (np.arange(n * ss) + 0.5) * res / ss
        X, Y = np.meshgrid(sub, sub)                               # (n*ss, n*ss) x then y
        occ = np.zeros_like(X, bool)
        for o in self.objects:
            if o["name"].startswith("wall"):
                continue
            if o["kind"] == "box":
                occ |= (np.abs(X - o["cx"]) <= o["sx"] / 2) & (np.abs(Y - o["cy"]) <= o["sy"] / 2)
            else:
                occ |= (X - o["cx"]) ** 2 + (Y - o["cy"]) ** 2 <= o["r"] ** 2
        grid = occ.reshape(n, ss, n, ss).any(axis=(1, 3))          # [row=y, col=x]
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
        return grid


def cell_of(x, y, res=C.RES):
    return int(np.floor(y / res)), int(np.floor(x / res))          # (row, col)


def center_of(r, c, res=C.RES):
    return (c + 0.5) * res, (r + 0.5) * res                        # (x, y)
