import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle, Circle

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.titlesize": 11})

COLORS = {"orange": "#e07b00", "blue": "#1f3fff", "green": "#16a34a", "magenta": "#d11fd1", "gray": "#666666"}


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


def save_table(df, name, floatfmt=3):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    with open(os.path.join(TAB, name + ".md"), "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=f".{floatfmt}g"))
    return df


def draw_world(ax, world, alpha=1.0, label_landmarks=False, low_hatch=True):
    for o in world.objects:
        col = COLORS[o["color"]]
        if o["kind"] == "box":
            if o["name"].startswith("wall"):
                continue
            hatch = "///" if (low_hatch and o["h"] < 0.2) else None
            ax.add_patch(Rectangle((o["cx"] - o["sx"] / 2, o["cy"] - o["sy"] / 2), o["sx"], o["sy"],
                                   fc=col, ec="k", lw=0.6, alpha=alpha, hatch=hatch, zorder=2))
        else:
            ax.add_patch(Circle((o["cx"], o["cy"]), o["r"], fc=col, ec="k", lw=0.6, alpha=alpha, zorder=2))
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.2, 10.2); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def grid_extent():
    return (0, 10, 0, 10)


def show_grid(ax, occ, cmap="Greys", alpha=0.85):
    ax.imshow(occ, origin="lower", extent=grid_extent(), cmap=cmap, alpha=alpha, vmin=0, vmax=1, zorder=1)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def step_metrics(t, y, ref, y0, settle_band=0.02, tail=1.0):
    """Standard step-response metrics, normalised so the initial value is 0 and the reference is 1."""
    A = ref - y0
    n = (np.asarray(y) - y0) / A
    t = np.asarray(t)
    os_ = max(0.0, n.max() - 1.0) * 100
    def first(th):
        i = np.nonzero(n >= th)[0]
        return t[i[0]] if len(i) else np.nan
    rise = first(0.9) - first(0.1) if not np.isnan(first(0.9)) else np.nan
    out = np.nonzero(np.abs(n - 1) > settle_band)[0]
    settle = (t[out[-1]] if len(out) else 0.0)
    if len(out) and out[-1] >= len(t) - 2:
        settle = np.nan                            # never settled inside the window
    m = t >= t[-1] - tail
    sse = float((ref - np.asarray(y)[m].mean()))
    itae = float(np.trapezoid(t * np.abs(1 - n), t) / (t[-1] ** 2))
    return dict(overshoot_pct=os_, rise_s=rise, settle_s=settle, sse=sse, itae=itae)


def load_gains():
    g = json.load(open(os.path.join(ROOT, "navlib", "tuned_gains.json")))
    return (g["Kp"], g["Ki"], g["Kd"])


def inflate_bool(occ, inflate):
    import cv2
    from navlib import config as C
    k = int(math.ceil(inflate / C.RES))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = ((xx ** 2 + yy ** 2) * C.RES ** 2 <= inflate ** 2 + 1e-9).astype(np.uint8)
    return cv2.dilate(occ.astype(np.uint8), disk).astype(bool)


def confusion(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec,
                F1=2 * prec * rec / max(prec + rec, 1e-9), IoU=tp / max(tp + fp + fn, 1))


# portable (Windows/Linux/macOS) location for the cached survey map shared between experiment stages
CACHE_DIR = os.path.join(ROOT, "results", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
SURVEY_PKL = os.path.join(CACHE_DIR, "survey_seed0.pkl")
