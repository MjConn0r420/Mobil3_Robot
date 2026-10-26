from common import *
import pickle
from navlib import config as C
from navlib.world import World
from navlib.sim import run_survey, plan_on_grid, track, tracking_stats
from navlib.planning import path_length, cells_to_xy, resample

world = World()
GAINS = load_gains()
lb = [o for o in world.objects if o["name"] == "low_box"][0]


def full_pipeline(seed, mode="fused", algo="astar", smooth=True, lidar_sigma=C.LIDAR_SIGMA,
                  pix_sigma=C.CAM_PIX_SIGMA, inflate=C.INFLATE, v_max=C.V_MAX, grid=None, cam_h_ratio=1.0):
    import navlib.perception as P
    if grid is None:
        P.CAM_H_BELIEF = C.CAM_HEIGHT * cam_h_ratio
        try:
            grid, _, _ = run_survey(world, seed=seed, gains=GAINS, lidar_sigma=lidar_sigma, pix_sigma=pix_sigma)
        finally:
            P.CAM_H_BELIEF = C.CAM_HEIGHT
    blocked = grid.planning_grid(mode, inflate)
    res, raw, sm = plan_on_grid(blocked, C.START, C.GOAL, algo)
    if res["found"] is False:
        return dict(ok=False, grid=grid, plan=res)
    path = sm if smooth else raw
    tr = track(world, path, GAINS, seed=100 + seed, v_max=v_max)
    return dict(ok=True, grid=grid, blocked=blocked, plan=res, raw=raw, smooth=sm, path=path, track=tr,
                stats=tracking_stats(tr, path))


def run():
    grid0 = pickle.load(open(SURVEY_PKL, "rb"))["grid"]
    # ---------------- main closed-loop run + variants (seed 0 map) ----------------
    variants = [("A* + smoothing (proposed)", dict(algo="astar", smooth=True, mode="fused")),
                ("Dijkstra + smoothing", dict(algo="dijkstra", smooth=True, mode="fused")),
                ("A* raw grid path (no smoothing)", dict(algo="astar", smooth=False, mode="fused")),
                ("A* + smoothing, LiDAR-only map", dict(algo="astar", smooth=True, mode="lidar")),
                ("A* + smoothing, camera-only map", dict(algo="astar", smooth=True, mode="camera"))]
    runs, rows = {}, []
    for name, kw in variants:
        r = full_pipeline(0, grid=grid0, **kw)
        runs[name] = r
        if not r["ok"]:
            rows.append(dict(configuration=name, reached=False)); continue
        rows.append(dict(configuration=name, **r["stats"], nodes=r["plan"]["expanded"],
                         plan_ms=r["plan"]["time"] * 1000))
    tab = pd.DataFrame(rows); save_table(tab.round(3), "integration_performance_seed0")

    # repeated-noise statistics for the proposed pipeline (10 tracking seeds on the seed-0 map)
    rep = []
    r0 = runs["A* + smoothing (proposed)"]
    for s in range(10):
        tr = track(world, r0["path"], GAINS, seed=200 + s)
        rep.append(tracking_stats(tr, r0["path"]))
    rdf = pd.DataFrame(rep)
    summ = rdf.agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    summ = summ[summ.metric.isin(["time_s", "planned_len", "actual_len", "cte_mean", "cte_rms", "cte_max", "min_clear"])]
    save_table(summ.round(4), "integration_repeatability_10_runs", floatfmt=4)
    print("success rate 10 runs:", rdf.reached.mean(), "collisions:", rdf.collided.sum())

    # ---------------- fig12: planned vs actual ----------------
    r = r0; L = r["track"]["log"]
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    show_grid(ax, r["blocked"].astype(float), cmap="Greys", alpha=0.35); draw_world(ax, world, alpha=0.5)
    raw = r["raw"]; ax.plot(raw[:, 0], raw[:, 1], color="tab:cyan", lw=1.5, label="A* grid path")
    ax.plot(r["smooth"][:, 0], r["smooth"][:, 1], "o--", color="k", lw=1.5, ms=5, label="smoothed plan (waypoints)")
    ax.plot(L[:, 1], L[:, 2], color="red", lw=2, label="actual (closed loop)")
    ax.plot(*C.START, "go", ms=11, zorder=6); ax.plot(*C.GOAL, "m*", ms=16, zorder=6)
    ax.set_title("Planned vs actual trajectory (fused map, A* + PID)"); ax.legend(loc="lower right", fontsize=8)
    a1 = fig.add_subplot(gs[0, 1]); a1.plot(L[:, 0], L[:, 7] * 100, color="C3"); a1.set(title="Cross-track error", xlabel="t [s]", ylabel="[cm]")
    a2 = fig.add_subplot(gs[0, 2]); a2.plot(L[:, 0], np.degrees(L[:, 6]), color="C2"); a2.set(title="Heading error to look-ahead point", xlabel="t [s]", ylabel="[deg]")
    a3 = fig.add_subplot(gs[1, 1]); a3.plot(L[:, 0], L[:, 4], label="v"); a3.plot(L[:, 0], L[:, 5], label="w", lw=0.8)
    a3.set(title="Executed velocities", xlabel="t [s]", ylabel="m/s | rad/s"); a3.legend()
    a4 = fig.add_subplot(gs[1, 2]); a4.hist(L[:, 7] * 100, bins=40, color="C3", alpha=0.8); a4.set(title="Cross-track error distribution", xlabel="[cm]")
    fig.tight_layout(); savefig(fig, "fig12_trajectory_planned_vs_actual.png")

    # ---------------- fig13: ablation (what fusion buys) ----------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.8))
    for a, name in zip(ax, ["A* + smoothing (proposed)", "A* + smoothing, LiDAR-only map", "A* + smoothing, camera-only map"]):
        rr = runs[name]
        if not rr["ok"]:
            a.set_title(name + ": no path"); continue
        show_grid(a, rr["blocked"].astype(float), cmap="Greys", alpha=0.35); draw_world(a, world, alpha=0.5)
        a.plot(rr["smooth"][:, 0], rr["smooth"][:, 1], "k--", lw=1.5, label="plan")
        LL = rr["track"]["log"]; a.plot(LL[:, 1], LL[:, 2], "r-", lw=2, label="actual")
        if rr["track"]["collided"]:
            a.plot(LL[-1, 1], LL[-1, 2], "X", color="red", ms=16, mec="k", zorder=9, label="COLLISION")
        st = rr["stats"]
        a.set_title(f"{name}\nreached={st['reached']}  collided={st['collided']}  min clearance={st['min_clear']:.2f} m")
        a.legend(loc="lower right", fontsize=8)
    savefig(fig, "fig13_fusion_ablation.png")

    # ---------------- sweeps ----------------
    gt_inf = inflate_bool(world.gt_occupancy(), C.INFLATE)
    truth = {o["color"]: np.array([o["cx"], o["cy"]]) for o in world.objects if o["kind"] == "cyl"}

    def lm_err(grid):
        lm = grid.landmark_positions()
        e = [np.hypot(*(p - truth[c])) for c, p in lm.items()]
        return float(np.mean(e) * 100) if e else np.nan

    def agg_row(rr, **kw):
        cm = confusion(inflate_bool(rr["grid"].occupied("fused"), C.INFLATE), gt_inf)
        row = dict(kw, F1=cm["F1"], recall=cm["recall"], precision=cm["precision"], lm_err_cm=lm_err(rr["grid"]))
        if rr["ok"]:
            st = rr["stats"]
            row.update(reached=st["reached"], collided=st["collided"], path_len=st["planned_len"], time_s=st["time_s"],
                       cte_rms=st["cte_rms"], min_clear=st["min_clear"])
        else:
            row.update(reached=False, collided=False)
        return row

    # (a) sensor noise (new survey/map per run; 3 seeds)
    rows = []
    for ls, ps, lab in [(0.01, 6, "1x"), (0.05, 25, "5x"), (0.10, 50, "10x"), (0.20, 90, "20x"), (0.40, 140, "40x")]:
        for seed in range(3):
            rows.append(agg_row(full_pipeline(seed, lidar_sigma=ls, pix_sigma=ps), noise=lab, lidar_sigma_m=ls, pix_sigma=ps, seed=seed))
    ns = pd.DataFrame(rows); save_table(ns.round(3), "sweep_sensor_noise_raw")
    nsa = ns.groupby("noise", sort=False).agg(lidar_sigma_m=("lidar_sigma_m", "first"), pix_sigma=("pix_sigma", "first"), F1=("F1", "mean"), recall=("recall", "mean"),
                                              precision=("precision", "mean"), lm_err_cm=("lm_err_cm", "mean"), path_len=("path_len", "mean"),
                                              cte_rms=("cte_rms", "mean"), min_clear=("min_clear", "mean"), success=("reached", "mean"),
                                              collided=("collided", "mean")).reset_index()
    save_table(nsa.round(3), "sweep_sensor_noise")

    # (b) camera height mis-calibration (systematic error, 3 seeds)
    rows = []
    for ratio in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
        for seed in range(3):
            rows.append(agg_row(full_pipeline(seed, cam_h_ratio=ratio), height_error_pct=(ratio - 1) * 100, seed=seed))
    cal = pd.DataFrame(rows)
    cala = cal.groupby("height_error_pct").agg(F1=("F1", "mean"), recall=("recall", "mean"), precision=("precision", "mean"), lm_err_cm=("lm_err_cm", "mean"),
                                               path_len=("path_len", "mean"), cte_rms=("cte_rms", "mean"), min_clear=("min_clear", "mean"),
                                               success=("reached", "mean"), collided=("collided", "mean")).reset_index()
    save_table(cala.round(3), "sweep_camera_calibration")

    # (c) inflation radius (fixed map)
    rows = []
    for infl in [0.17, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
        for seed in range(3):
            rr = full_pipeline(seed, grid=grid0, inflate=infl)
            rows.append(dict(inflate=infl, seed=seed, **rr["stats"]) if rr["ok"] else dict(inflate=infl, seed=seed, reached=False, collided=False))
    inf_df = pd.DataFrame(rows)
    inf_a = inf_df.groupby("inflate").agg(path_len=("planned_len", "mean"), time_s=("time_s", "mean"), min_clear=("min_clear", "mean"),
                                           success=("reached", "mean"), collided=("collided", "mean")).reset_index()
    save_table(inf_a.round(3), "sweep_inflation")

    # (d) speed, with and without the actuator disturbance
    rows = []
    for v in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for dist, (wb, eps) in {"with disturbance": (0.03, 0.01), "no disturbance": (0.0, 0.0)}.items():
            for seed in range(5):
                tr = track(world, r0["path"], GAINS, seed=300 + seed, v_max=v, w_bias=wb, eps=eps)
                rows.append(dict(v_max=v, case=dist, seed=seed, **tracking_stats(tr, r0["path"])))
    sp = pd.DataFrame(rows)
    sp_a = sp.groupby(["case", "v_max"]).agg(time_s=("time_s", "mean"), cte_rms=("cte_rms", "mean"), cte_max=("cte_max", "mean"),
                                              min_clear=("min_clear", "mean"), success=("reached", "mean")).reset_index()
    sp_a["brake_dist_m"] = sp_a["v_max"] ** 2 / (2 * 0.6)
    save_table(sp_a.round(3), "sweep_speed")

    # (e) look-ahead distance x speed (corner cutting is geometric)
    rows = []
    for L_ in [0.2, 0.3, 0.4, 0.6, 0.8]:
        for v in [0.2, 0.4, 0.6]:
            for seed in range(3):
                tr = track(world, r0["path"], GAINS, seed=400 + seed, v_max=v, lookahead=L_)
                rows.append(dict(lookahead=L_, v_max=v, seed=seed, **tracking_stats(tr, r0["path"])))
    la = pd.DataFrame(rows).groupby(["lookahead", "v_max"]).agg(cte_rms=("cte_rms", "mean"), cte_max=("cte_max", "mean"),
                                                               time_s=("time_s", "mean"), min_clear=("min_clear", "mean"), success=("reached", "mean")).reset_index()
    save_table(la.round(3), "sweep_lookahead_speed")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for v in [0.2, 0.4, 0.6]:
        s2 = la[la.v_max == v]
        ax[0].plot(s2["lookahead"], s2["cte_max"] * 100, "o-", label=f"v_max={v}")
        ax[1].plot(s2["lookahead"], s2["cte_rms"] * 100, "o-", label=f"v_max={v}")
    ax[0].set(title="Max cross-track error vs look-ahead", xlabel="look-ahead [m]", ylabel="[cm]"); ax[1].set(title="RMS cross-track error vs look-ahead", xlabel="look-ahead [m]", ylabel="[cm]")
    ax[0].legend(); fig.tight_layout(); savefig(fig, "fig15_lookahead_speed.png")

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]; x = np.arange(len(nsa))
    a.plot(x, nsa["F1"], "o-", label="map F1"); a.plot(x, nsa["recall"], "s--", label="map recall"); a.plot(x, nsa["precision"], "^:", label="map precision")
    a.set_xticks(x); a.set_xticklabels([f"{l}\n({s:.2f} m, {p:.0f})" for l, s, p in zip(nsa["noise"], nsa["lidar_sigma_m"], nsa["pix_sigma"])], fontsize=8)
    a.set(title="(a) Random sensor noise: LiDAR sigma [m], pixel sigma", ylabel="planning-map score", ylim=(0.4, 1.02)); a.legend(fontsize=8, loc="lower left")
    b2 = a.twinx(); b2.plot(x, nsa["min_clear"] * 100, "D-", color="C3"); b2.set_ylabel("min clearance [cm]", color="C3"); b2.grid(False)
    a = ax[0, 1]; a.plot(cala["height_error_pct"], cala["F1"], "o-", label="map F1"); a.plot(cala["height_error_pct"], cala["recall"], "s--", label="map recall")
    a.set(title="(b) Systematic error: camera-height mis-calibration", xlabel="assumed camera height error [%]", ylabel="planning-map score", ylim=(0.4, 1.02)); a.legend(fontsize=8, loc="lower left")
    b2 = a.twinx(); b2.plot(cala["height_error_pct"], cala["lm_err_cm"], "D-", color="C3"); b2.set_ylabel("landmark error [cm]", color="C3"); b2.grid(False)
    for xx_, s_, c_ in zip(cala["height_error_pct"], cala["success"], cala["collided"]):
        if c_ > 0:
            a.annotate(f"collide {c_*100:.0f}%", (xx_, 0.45), fontsize=7, rotation=90, color="red")
    a = ax[1, 0]; a.plot(inf_a["inflate"], inf_a["path_len"], "o-"); a.set(title="(c) Planning: inflation radius", xlabel="inflation radius [m]", ylabel="planned path length [m]")
    b2 = a.twinx(); b2.plot(inf_a["inflate"], inf_a["min_clear"] * 100, "s-", color="C3"); b2.set_ylabel("min clearance achieved [cm]", color="C3"); b2.grid(False)
    for xx_, s_, c_ in zip(inf_a["inflate"], inf_a["success"], inf_a["collided"]):
        if c_ > 0:
            a.annotate("collision", (xx_, a.get_ylim()[0] + 0.05), color="red", fontsize=8, rotation=90)
        if s_ == 0 and c_ == 0:
            a.annotate("no path", (xx_, a.get_ylim()[0] + 0.05), color="gray", fontsize=8, rotation=90)
    a = ax[1, 1]
    for case, ls in [("with disturbance", "-"), ("no disturbance", "--")]:
        s2 = sp_a[sp_a.case == case]; a.plot(s2["v_max"], s2["cte_rms"] * 100, "o" + ls, label=f"RMS CTE, {case}")
    a.set(title="(d) Control: speed vs tracking error (look-ahead 0.4 m)", xlabel="v_max [m/s]", ylabel="RMS cross-track error [cm]"); a.legend(fontsize=8)
    b2 = a.twinx(); s2 = sp_a[sp_a.case == "no disturbance"]; b2.plot(s2["v_max"], s2["time_s"], "D-", color="C2"); b2.set_ylabel("time-to-goal [s]", color="C2"); b2.grid(False)
    fig.tight_layout(); savefig(fig, "fig14_tradeoffs.png")
    return tab, summ, nsa, cala, inf_a, sp_a


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for t in out:
        print(t.round(3).to_string()); print()
