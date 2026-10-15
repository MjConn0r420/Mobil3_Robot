from common import *
import pickle, time, cv2
from navlib import config as C
from navlib.world import World, cell_of
from navlib.planning import astar, dijkstra, cells_to_xy, smooth_path, path_length

ALGOS = [("Dijkstra", "dijkstra", None), ("A* (octile)", "astar", "octile"),
         ("A* (Euclidean)", "astar", "euclid"), ("A* (Manhattan)", "astar", "manhattan")]


def solve(blocked, s, g, kind, h, reps=7):
    best = None; ts = []
    for _ in range(reps):
        r = dijkstra(blocked, s, g) if kind == "dijkstra" else astar(blocked, s, g, h)
        ts.append(r["time"]); best = r
    best["time_ms"] = float(np.median(ts) * 1000)
    return best


def run():
    world = World()
    grid = pickle.load(open(SURVEY_PKL, "rb"))["grid"]
    blocked = grid.planning_grid("fused")
    S, G = cell_of(*C.START), cell_of(*C.GOAL)

    # ---- main problem
    res = {name: solve(blocked, S, G, k, h) for name, k, h in ALGOS}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.8))
    for a, name in zip(ax[:2], ["Dijkstra", "A* (octile)"]):
        r = res[name]
        exp = np.where(r["closed"], 1.0, np.nan)
        show_grid(a, blocked.astype(float), cmap="Greys", alpha=0.55)
        a.imshow(exp, origin="lower", extent=grid_extent(), cmap="YlGnBu", alpha=0.55, vmin=0, vmax=1.3, zorder=2)
        xy = cells_to_xy(r["path"]); a.plot(xy[:, 0], xy[:, 1], "r-", lw=2, zorder=5)
        a.plot(*C.START, "go", ms=10, zorder=6); a.plot(*C.GOAL, "m*", ms=15, zorder=6)
        a.set_title(f"{name}: {r['expanded']} nodes expanded, cost {r['cost']:.2f} m, {r['time_ms']:.1f} ms")
    # overlay of all algorithms' paths
    show_grid(ax[2], blocked.astype(float), cmap="Greys", alpha=0.55)
    sty = {"Dijkstra": ("k", "-", 3), "A* (octile)": ("r", "--", 2), "A* (Euclidean)": ("b", ":", 2), "A* (Manhattan)": ("g", "-.", 2)}
    for name, r in res.items():
        xy = cells_to_xy(r["path"]); c, ls, lw = sty[name]
        ax[2].plot(xy[:, 0], xy[:, 1], color=c, ls=ls, lw=lw, label=f"{name}: {r['cost']:.2f} m")
    sm = smooth_path(cells_to_xy(res["A* (octile)"]["path"]), blocked)
    ax[2].plot(sm[:, 0], sm[:, 1], "o-", color="orange", lw=1.5, ms=4, label=f"smoothed A*: {path_length(sm):.2f} m")
    ax[2].plot(*C.START, "go", ms=10); ax[2].plot(*C.GOAL, "m*", ms=15)
    ax[2].legend(fontsize=8, loc="lower right"); ax[2].set_title("Paths on the fused, inflated map")
    savefig(fig, "fig06_planner_paths_explored.png")

    # ---- multi-pair statistics
    rng = np.random.default_rng(11)
    free = np.argwhere(~blocked)
    pairs = []
    while len(pairs) < 20:
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        if np.hypot(*(a - b)) * C.RES > 5.0:
            pairs.append((tuple(a), tuple(b)))
    pairs.insert(0, (S, G))
    rows = []
    for pi, (s, g) in enumerate(pairs):
        base = None
        for name, k, h in ALGOS:
            r = solve(blocked, s, g, k, h, reps=3)
            if name == "Dijkstra":
                base = r["cost"]
            rows.append(dict(pair=pi, algo=name, found=r["found"], expanded=r["expanded"], time_ms=r["time_ms"],
                             cost_m=r["cost"], suboptimality_pct=(r["cost"] / base - 1) * 100 if base else 0))
    df = pd.DataFrame(rows)
    agg = df.groupby("algo").agg(nodes_mean=("expanded", "mean"), nodes_std=("expanded", "std"),
                                 time_ms_mean=("time_ms", "mean"), time_ms_std=("time_ms", "std"),
                                 path_m_mean=("cost_m", "mean"), subopt_pct_mean=("suboptimality_pct", "mean"),
                                 subopt_pct_max=("suboptimality_pct", "max"), n_suboptimal=("suboptimality_pct", lambda x: int((x > 1e-6).sum()))).reindex([a[0] for a in ALGOS]).reset_index()
    base_nodes = agg.loc[agg.algo == "Dijkstra", "nodes_mean"].iloc[0]
    agg["nodes_vs_dijkstra"] = agg["nodes_mean"] / base_nodes
    save_table(agg.round(3), "planner_comparison_21_pairs")
    main = pd.DataFrame([dict(algo=n, nodes_expanded=res[n]["expanded"], time_ms=res[n]["time_ms"], path_cost_m=res[n]["cost"],
                              waypoints=len(res[n]["path"])) for n, _, _ in ALGOS])
    save_table(main.round(3), "planner_comparison_main_problem")

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    cols = sns.color_palette("Set2", 4)
    sns.boxplot(data=df, x="algo", y="expanded", ax=ax[0], palette=cols, order=[a[0] for a in ALGOS], hue="algo", legend=False)
    ax[0].set(title="Nodes expanded (21 start/goal pairs)", xlabel="", ylabel="nodes")
    sns.boxplot(data=df, x="algo", y="time_ms", ax=ax[1], palette=cols, order=[a[0] for a in ALGOS], hue="algo", legend=False)
    ax[1].set(title="Planning time", xlabel="", ylabel="ms")
    sns.barplot(data=df, x="algo", y="suboptimality_pct", ax=ax[2], palette=cols, order=[a[0] for a in ALGOS], hue="algo", legend=False)
    ax[2].set(title="Path suboptimality vs Dijkstra", xlabel="", ylabel="% longer")
    for a in ax[:3]:
        a.tick_params(axis="x", rotation=25)
    # ---- scaling with resolution (ground-truth map, same task)
    gt = world.gt_occupancy(0.025, ss=2)
    rows2 = []
    _w = np.zeros((20, 20), bool); dijkstra(_w, (1, 1), (18, 18)); astar(_w, (1, 1), (18, 18))   # warm-up
    for res_m in [0.2, 0.1, 0.05, 0.025]:
        n = int(round(10 / res_m))
        g_occ = world.gt_occupancy(res_m, ss=3 if res_m >= 0.05 else 2)
        k = int(math.ceil(C.INFLATE / res_m)); yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = ((xx ** 2 + yy ** 2) * res_m ** 2 <= C.INFLATE ** 2 + 1e-9).astype(np.uint8)
        bl = cv2.dilate(g_occ.astype(np.uint8), disk).astype(bool)
        s_ = (int(C.START[1] / res_m), int(C.START[0] / res_m)); g_ = (int(C.GOAL[1] / res_m), int(C.GOAL[0] / res_m))
        for name, kind, h in [ALGOS[0], ALGOS[1]]:
            t0 = time.perf_counter()
            r = dijkstra(bl, s_, g_) if kind == "dijkstra" else astar(bl, s_, g_, h)
            rows2.append(dict(res_m=res_m, cells=n * n, algo=name, expanded=r["expanded"], time_ms=(time.perf_counter() - t0) * 1000))
    d2 = pd.DataFrame(rows2); save_table(d2.round(4), "planner_scaling_resolution", floatfmt=4)
    for name in ["Dijkstra", "A* (octile)"]:
        s2 = d2[d2.algo == name]
        ax[3].plot(s2["cells"], s2["time_ms"], "o-", label=name)
    ax[3].set(xscale="log", yscale="log", title="Scaling with map size (grid resolution 0.2 -> 0.025 m)", xlabel="grid cells", ylabel="time [ms]")
    ax[3].legend()
    fig.tight_layout(); savefig(fig, "fig07_planner_comparison.png")
    return agg, main, d2, res


if __name__ == "__main__":
    agg, main, d2, res = run()
    pd.set_option("display.width", 250)
    print(main.round(3).to_string()); print(agg.round(3).to_string()); print(d2.round(1).to_string())
