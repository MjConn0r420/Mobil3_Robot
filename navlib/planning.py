"""Grid path planning: Dijkstra and A* (8-connected, no corner cutting), path smoothing."""
import heapq, math, time
import numpy as np
from . import config as C

SQRT2 = math.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]   # (dr, dc, cost)


def heuristic(name, r, c, gr, gc):
    dr, dc = abs(r - gr), abs(c - gc)
    if name == "euclid":
        return math.hypot(dr, dc)
    if name == "manhattan":                       # NOT admissible with diagonal moves
        return float(dr + dc)
    if name == "octile":                          # exact cost on an empty 8-connected grid
        return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)
    return 0.0


def _neighbours(blocked, r, c):
    n = blocked.shape[0]
    for dr, dc, cost in MOVES:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < n and 0 <= cc < n) or blocked[rr, cc]:
            continue
        if dr != 0 and dc != 0 and (blocked[r + dr, c] or blocked[r, c + dc]):
            continue                              # forbid squeezing diagonally between two obstacles
        yield rr, cc, cost


def _search(blocked, start, goal, hname):
    t0 = time.perf_counter()
    n = blocked.shape[0]
    g = np.full((n, n), np.inf)
    parent = {}
    closed = np.zeros((n, n), bool)
    g[start] = 0.0
    h0 = heuristic(hname, *start, *goal)
    pq = [(h0, h0, start)]
    expanded = 0
    found = False
    while pq:
        f, _, (r, c) = heapq.heappop(pq)
        if closed[r, c]:
            continue                              # stale queue entry
        closed[r, c] = True
        expanded += 1
        if (r, c) == goal:
            found = True
            break
        for rr, cc, cost in _neighbours(blocked, r, c):
            ng = g[r, c] + cost
            if ng < g[rr, cc]:
                g[rr, cc] = ng
                parent[(rr, cc)] = (r, c)
                h = heuristic(hname, rr, cc, *goal)
                heapq.heappush(pq, (ng + h, h, (rr, cc)))
    dt = time.perf_counter() - t0
    path = []
    if found:
        cur = goal
        while cur != start:
            path.append(cur); cur = parent[cur]
        path.append(start); path.reverse()
    return dict(found=found, path=path, cost=float(g[goal]) * C.RES if found else math.inf,
                expanded=expanded, closed=closed, time=dt)


def dijkstra(blocked, start, goal):
    """Uniform-cost search = A* with h == 0."""
    return _search(blocked, start, goal, "zero")


def astar(blocked, start, goal, h="octile"):
    return _search(blocked, start, goal, h)


# ------------------------------------------------------------------ geometry helpers
def cells_to_xy(cells):
    return np.array([[(c + 0.5) * C.RES, (r + 0.5) * C.RES] for r, c in cells])


def line_free(blocked, p, q):
    """Line-of-sight test between two world points on the (inflated) grid, sampled every RES/4."""
    d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
    k = max(2, int(d / (C.RES / 4)))
    xs, ys = np.linspace(p[0], q[0], k), np.linspace(p[1], q[1], k)
    cs = np.clip((xs / C.RES).astype(int), 0, blocked.shape[1] - 1)
    rs = np.clip((ys / C.RES).astype(int), 0, blocked.shape[0] - 1)
    return not blocked[rs, cs].any()


def smooth_path(xy, blocked):
    """Greedy line-of-sight shortcutting: removes the staircase of 8-connected grid paths
    (fewer heading changes for the controller, shorter path) while staying collision-free."""
    out = [xy[0]]
    i = 0
    while i < len(xy) - 1:
        j = len(xy) - 1
        while j > i + 1 and not line_free(blocked, xy[i], xy[j]):
            j -= 1
        out.append(xy[j]); i = j
    return np.array(out)


def resample(xy, spacing=0.05):
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], spacing)
    ss = np.append(ss, s[-1])
    return np.stack([np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])], 1)


def path_length(xy):
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())
