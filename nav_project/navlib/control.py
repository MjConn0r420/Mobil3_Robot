"""PID controller and the path-tracking law for a differential-drive (unicycle) robot.

Kinematic model            x' = v cos(th),  y' = v sin(th),  th' = w
Wheel mapping              w_R = (v + w*b/2)/r ,  w_L = (v - w*b/2)/r
Actuator dynamics (sim)    first-order lag tau on each wheel + 2-step delay + saturation
"""
import math
import numpy as np
from . import config as C


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PID:
    """Parallel-form PID with filtered derivative, output saturation and
    conditional-integration anti-windup (integration pauses while the output is
    saturated AND the error would push it further into saturation)."""

    def __init__(self, kp, ki, kd, umin=-C.W_MAX, umax=C.W_MAX, tau_d=0.05, anti_windup=True):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.umin, self.umax, self.tau_d, self.aw = umin, umax, tau_d, anti_windup
        self.reset()

    def reset(self):
        self.I, self.e_prev, self.d_f, self.first = 0.0, 0.0, 0.0, True

    def update(self, e, dt):
        if self.first:                      # no derivative kick on the first sample
            self.e_prev, self.first = e, False
        de = (e - self.e_prev) / dt
        self.d_f += (de - self.d_f) * dt / (self.tau_d + dt)     # 1st-order low-pass on D
        self.e_prev = e
        I_c = self.I + e * dt
        u_unsat = self.kp * e + self.ki * I_c + self.kd * self.d_f
        u = min(max(u_unsat, self.umin), self.umax)
        if (not self.aw) or u == u_unsat or (e * u_unsat < 0):
            self.I = I_c
        return u


class PathTracker:
    """Lookahead path follower. Heading error towards a point `lookahead` metres ahead of the
    closest path point drives a heading PID (so cross-track error is corrected implicitly);
    forward speed follows a trapezoidal profile scaled by alignment."""

    def __init__(self, path_xy, gains, v_max=C.V_MAX, lookahead=0.40, a_dec=0.6,
                 goal_tol=0.08, anti_windup=True):
        self.path = np.asarray(path_xy, float)
        seg = np.hypot(*np.diff(self.path, axis=0).T)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.L, self.v_max, self.a_dec, self.tol = lookahead, v_max, a_dec, goal_tol
        self.pid = PID(*gains, anti_windup=anti_windup)
        self.i_near = 0
        self.goal = self.path[-1]

    def _closest(self, p):
        lo, hi = self.i_near, min(len(self.path), self.i_near + 60)
        d = np.hypot(*(self.path[lo:hi] - p).T)
        self.i_near = lo + int(d.argmin())
        return self.i_near

    def target(self, p):
        i = self._closest(p)
        s_t = min(self.s[i] + self.L, self.s[-1])
        j = int(np.searchsorted(self.s, s_t))
        return self.path[min(j, len(self.path) - 1)]

    def command(self, pose, dt):
        x, y, th = pose
        p = np.array([x, y])
        dist_goal = float(np.hypot(*(self.goal - p)))
        if dist_goal < self.tol:
            return 0.0, 0.0, True, 0.0
        tg = self.target(p)
        e = wrap(math.atan2(tg[1] - y, tg[0] - x) - th)
        w = self.pid.update(e, dt)
        v_prof = min(self.v_max, math.sqrt(2 * self.a_dec * dist_goal))
        v = v_prof * max(0.0, math.cos(e)) ** 2
        return v, w, False, e


class TurnInPlace:
    def __init__(self, gains, theta_ref):
        self.pid = PID(*gains)
        self.ref = theta_ref

    def command(self, pose, dt):
        e = wrap(self.ref - pose[2])
        w = self.pid.update(e, dt)
        done = abs(e) < 0.05 and abs(self.pid.d_f) < 0.3
        return 0.0, w, done, e


def wheel_speeds(v, w):
    wr = (v + w * C.TRACK / 2) / C.WHEEL_R
    wl = (v - w * C.TRACK / 2) / C.WHEEL_R
    s = max(1.0, abs(wr) / C.WHEEL_MAX, abs(wl) / C.WHEEL_MAX)   # scale both to keep curvature
    return wl / s, wr / s
