"""Turns the logs written by the Webots controller into the assignment's evaluation artefacts.

    python analysis/plot_webots_run.py                # uses logs/webots_*.{csv,npz}
    python analysis/plot_webots_run.py --prefix mock  # same, for the mock-API test run

Outputs (results/webots/):  webots_trajectory.png, webots_performance.csv / .md"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from navlib.world import World
from navlib.planning import path_length

ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="webots"); a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
logs, out = os.path.join(root, "logs"), os.path.join(root, "results", "webots")
os.makedirs(out, exist_ok=True)
tr = pd.read_csv(os.path.join(logs, f"{a.prefix}_trajectory.csv"))
plan = np.loadtxt(os.path.join(logs, f"{a.prefix}_plan.csv"), delimiter=",", skiprows=1)
g = np.load(os.path.join(logs, f"{a.prefix}_grid.npz"))
world = World()

actual = float(np.hypot(np.diff(tr.x), np.diff(tr.y)).sum())
clear = np.array([world.clearance(x, y) for x, y in zip(tr.x, tr.y)])
perf = pd.DataFrame([dict(planned_path_m=path_length(plan), actual_path_m=actual, time_to_goal_s=tr.t.iloc[-1] - tr.t.iloc[0],
                          cte_mean_cm=tr.cross_track.mean() * 100, cte_rms_cm=np.sqrt((tr.cross_track ** 2).mean()) * 100,
                          cte_max_cm=tr.cross_track.max() * 100, min_clearance_m=clear.min(),
                          final_goal_error_m=float(np.hypot(tr.x.iloc[-1] - 9.0, tr.y.iloc[-1] - 9.0)))])
perf.round(3).to_csv(os.path.join(out, "webots_performance.csv"), index=False)
open(os.path.join(out, "webots_performance.md"), "w").write(perf.to_markdown(index=False, floatfmt=".3g"))
print(perf.round(3).T)

fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax[0].imshow(g["blocked"], origin="lower", extent=(0, 10, 0, 10), cmap="Greys", alpha=0.35)
for o in world.objects:
    if o["name"].startswith("wall"):
        continue
    if o["kind"] == "box":
        ax[0].add_patch(Rectangle((o["cx"] - o["sx"] / 2, o["cy"] - o["sy"] / 2), o["sx"], o["sy"], fc="#e0a050", ec="k", lw=0.5, alpha=0.6))
    else:
        ax[0].add_patch(Circle((o["cx"], o["cy"]), o["r"], fc="#8888ff", ec="k", lw=0.5, alpha=0.6))
ax[0].plot(g["raw"][:, 0], g["raw"][:, 1], color="tab:cyan", lw=1.2, label="A* grid path")
ax[0].plot(plan[:, 0], plan[:, 1], "ko--", ms=4, lw=1.2, label="smoothed plan")
ax[0].plot(tr.x, tr.y, "r-", lw=2, label="actual (Webots)")
ax[0].set(xlim=(-0.2, 10.2), ylim=(-0.2, 10.2), aspect="equal", xlabel="x [m]", ylabel="y [m]", title="Planned vs actual trajectory (Webots)")
ax[0].legend(loc="lower right", fontsize=8)
ax[1].plot(tr.t - tr.t.iloc[0], tr.cross_track * 100, "C3"); ax[1].set(title="Cross-track error", xlabel="t [s]", ylabel="cm")
ax[2].plot(tr.t - tr.t.iloc[0], np.degrees(tr.heading_err), "C2"); ax[2].set(title="Heading error", xlabel="t [s]", ylabel="deg")
fig.tight_layout(); fig.savefig(os.path.join(out, "webots_trajectory.png"), dpi=150)
print("wrote", out)
