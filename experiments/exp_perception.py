from common import *
import pickle, cv2
from navlib import config as C
from navlib.world import World, SURVEY_ROUTE
from navlib.sim import run_survey
from navlib.sensors import SimLidar, SimCamera, LIDAR_ANGLES
from navlib.perception import (preprocess_lidar, camera_masks, detect, contact_points, to_world,
                               HSV_RANGES)


def run():
    world = World()
    gains = load_gains()
    dets_log = []
    grid, sim, traj = run_survey(world, seed=0, gains=gains, log_detections=dets_log)
    pickle.dump(dict(grid=grid, traj=traj), open(SURVEY_PKL, "wb"))
    gt = world.gt_occupancy()
    gt_inf = inflate_bool(gt, C.INFLATE)

    # ------------------------------------------------- fig01: world + sensors at one pose
    pose = (1.0, 6.2, 0.45)
    rng = np.random.default_rng(5)
    lid, cam = SimLidar(world, rng), SimCamera(world, rng)
    raw = lid.scan(pose); img = cam.render(pose)
    fig = plt.figure(figsize=(14, 5.8))
    ax = fig.add_subplot(1, 2, 1)
    draw_world(ax, world)
    ax.plot(*zip(*[(s[0], s[1]) for s in SURVEY_ROUTE]), "--", color="gray", lw=1, label="survey route")
    ang = pose[2] + LIDAR_ANGLES; fin = np.isfinite(raw)
    ax.scatter(pose[0] + raw[fin] * np.cos(ang[fin]), pose[1] + raw[fin] * np.sin(ang[fin]), s=3, c="crimson", label="LiDAR returns", zorder=4)
    from matplotlib.patches import Wedge
    ax.add_patch(Wedge((pose[0] + C.CAM_XOFF * math.cos(pose[2]), pose[1] + C.CAM_XOFF * math.sin(pose[2])), 3.5,
                       math.degrees(pose[2] - C.CAM_FOV / 2), math.degrees(pose[2] + C.CAM_FOV / 2),
                       alpha=0.18, color="dodgerblue", label="camera FOV (usable range)"))
    ax.plot(*pose[:2], "k^", ms=10, zorder=5)
    lb = [o for o in world.objects if o["name"] == "low_box"][0]
    ax.annotate("low box (12 cm)\ninvisible to LiDAR", (lb["cx"], lb["cy"]), (3.9, 6.0), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_title("Arena, survey route and one sensing pose"); ax.legend(loc="lower right", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax2.set_title("Simulated camera frame (256x128, noisy)"); ax2.axis("off")
    savefig(fig, "fig01_world_and_sensors.png")

    # ------------------------------------------------- fig02: perception pipeline
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.5))
    ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0, 0].set_title("1. raw frame")
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ax[0, 1].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)); ax[0, 1].set_title("2. Gaussian blur 5x5")
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lo, hi = HSV_RANGES["obstacle"]
    m0 = cv2.inRange(hsv, np.array(lo), np.array(hi))
    ax[0, 2].imshow(m0, cmap="gray"); ax[0, 2].set_title("3. HSV threshold (orange): raw mask")
    masks, _ = camera_masks(img)
    ax[0, 3].imshow(masks["obstacle"] | masks["magenta"], cmap="gray"); ax[0, 3].set_title("4. open + close (noise removed)")
    ov = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    dets = detect(img)
    for d in dets:
        c = {"obstacle": (255, 255, 0), "magenta": (0, 255, 255), "blue": (255, 255, 0), "green": (255, 255, 0)}[d.cls]
        for u, v in zip(d.cols, d.rows):
            if 0 <= v < C.CAM_H:
                ov[max(v - 1, 0):v + 1, u] = c
        x, y, w, h = d.bbox
        cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(ov, d.cls, (x, max(y - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    ax[1, 0].imshow(ov); ax[1, 0].set_title("5. blobs + floor-contact boundary")
    pts = [to_world(contact_points(d), pose) for d in dets if len(contact_points(d))]
    draw_world(ax[1, 1], world, alpha=0.35)
    if pts:
        P = np.vstack(pts); ax[1, 1].scatter(P[:, 0], P[:, 1], s=4, c="k", zorder=5, label="camera contact points")
    ax[1, 1].plot(*pose[:2], "r^"); ax[1, 1].set_xlim(0, 7); ax[1, 1].set_ylim(4, 10); ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title("6. ground-plane back-projection")
    ranges = raw.copy(); clean = preprocess_lidar(raw)
    ax[1, 2].plot(np.degrees(LIDAR_ANGLES), np.where(np.isfinite(raw), raw, np.nan), ".", ms=3, label="raw")
    ax[1, 2].plot(np.degrees(LIDAR_ANGLES), np.where(np.isfinite(clean), clean, np.nan), "-", lw=1, label="median-filtered")
    ax[1, 2].set(title="LiDAR scan: spike/dropout rejection", xlabel="beam angle [deg]", ylabel="range [m]"); ax[1, 2].legend(fontsize=7)
    ax[1, 3].axis("off")
    ax[1, 3].text(0, 0.95, "Design decisions\n\n"
                  "- Blur before threshold: kills Gaussian pixel noise\n- HSV not RGB: hue is stable under shading\n"
                  "- Open (3x3): removes salt pixels\n- Close (5x5): heals pepper holes in blobs\n"
                  "- Area >= 40 px: rejects residual specks\n- Ground-plane range z=f*h/(v-cy):\n  works for ANY object height\n"
                  "- Points beyond 3.5 m dropped\n  (1 px = >5 % range error)\n- LiDAR 3-tap circular median:\n  removes 1-beam outliers, keeps edges",
                  va="top", fontsize=9, family="monospace")
    for a in ax[0]:
        a.axis("off")
    ax[1, 0].axis("off")
    fig.tight_layout(); savefig(fig, "fig02_perception_pipeline.png")

    # ------------------------------------------------- fig03: occupancy maps + metrics
    modes = ["lidar", "camera", "fused"]
    titles = {"lidar": "LiDAR only", "camera": "Camera only", "fused": "Fused (OR of log-odds)"}
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.3))
    rows = []
    for a, m in zip(ax[:3], modes):
        occ = grid.occupied(m)
        show_grid(a, occ.astype(float)); draw_world(a, world, alpha=0.25)
        inf = inflate_bool(occ, C.INFLATE)
        cm = confusion(inf, gt_inf)
        cm0 = confusion(occ, inflate_bool(gt, 0.1))        # raw (uninflated) vs GT dilated by 1 cell tolerance
        # recall specifically on the low box (inflated footprint)
        lbm = np.zeros_like(gt)
        yy, xx = np.mgrid[0:C.N, 0:C.N]
        cxs, cys = (xx + 0.5) * C.RES, (yy + 0.5) * C.RES
        lbm = (np.abs(cxs - lb["cx"]) <= lb["sx"] / 2 + 0.05) & (np.abs(cys - lb["cy"]) <= lb["sy"] / 2 + 0.05)
        low_rec = float((occ & lbm).sum() / max(1, lbm.sum()))
        edge = inflate_bool(lbm, 0.1) & ~lbm
        rows.append(dict(map=titles[m], planning_precision=cm["precision"], planning_recall=cm["recall"],
                         planning_F1=cm["F1"], planning_IoU=cm["IoU"], TP=cm["TP"], FP=cm["FP"], FN=cm["FN"], TN=cm["TN"],
                         lowbox_cells_marked=low_rec))
        a.set_title(f"{titles[m]}   F1(planning)={cm['F1']:.2f}")
    tr = pickle_load_traj()
    ax[0].plot(tr[:, 1], tr[:, 2], "-", lw=0.6, color="tab:blue", alpha=0.7, zorder=5)
    lm = grid.landmark_positions()
    for c_, p in lm.items():
        ax[2].plot(*p, "x", color="k", ms=8, mew=2, zorder=6)
    ax[3].imshow(inflate_bool(grid.occupied("fused"), C.INFLATE), origin="lower", extent=grid_extent(), cmap="Reds", alpha=0.6)
    ax[3].contour(np.linspace(0.05, 9.95, C.N), np.linspace(0.05, 9.95, C.N), gt_inf, levels=[0.5], colors="k", linewidths=1)
    ax[3].set_aspect("equal"); ax[3].set_title("Fused planning map (inflated 0.25 m)\nblack = ground-truth inflated boundary")
    savefig(fig, "fig03_occupancy_maps.png")
    df = pd.DataFrame(rows); save_table(df.round(3), "perception_map_quality")

    # ------------------------------------------------- fig04: confusion matrices (planning space)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for a, m in zip(ax, modes):
        inf = inflate_bool(grid.occupied(m), C.INFLATE)
        cm = confusion(inf, gt_inf)
        mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        norm = mat / mat.sum(1, keepdims=True)
        sns.heatmap(norm, annot=np.array([[f"{mat[i,j]}\n({norm[i,j]*100:.1f}%)" for j in range(2)] for i in range(2)]),
                    fmt="", cmap="Blues", cbar=False, ax=a, xticklabels=["free", "blocked"], yticklabels=["free", "blocked"], vmin=0, vmax=1)
        a.set_xlabel("predicted"); a.set_ylabel("ground truth"); a.set_title(f"{titles[m]}")
    fig.suptitle("Confusion matrices of the inflated (planning-space) map vs ground truth", y=1.03)
    savefig(fig, "fig04_confusion_matrices.png")

    # ------------------------------------------------- fig05: range accuracy + landmark error
    cam_err, cam_rng, lid_err, lid_rng = [], [], [], []
    for (p, dets_) in dets_log[::2]:
        org = np.array([p[0] + C.CAM_XOFF * math.cos(p[2]), p[1] + C.CAM_XOFF * math.sin(p[2])])
        for d in dets_:
            pr = contact_points(d)
            if len(pr) == 0:
                continue
            pw = to_world(pr, p)
            sel = np.arange(0, len(pw), 3)
            for q in pw[sel]:
                # error = distance to the true surface (clearance is distance to nearest obstacle surface)
                e = world.clearance(q[0], q[1])
                if q[0] < 0.15 or q[1] < 0.15 or q[0] > 9.85 or q[1] > 9.85:
                    continue
                cam_err.append(e); cam_rng.append(np.linalg.norm(q - org))
    rng2 = np.random.default_rng(9); lid2 = SimLidar(world, rng2)
    for (p, _) in dets_log[::8]:
        r = preprocess_lidar(lid2.scan(p)); fin = np.isfinite(r)
        a = p[2] + LIDAR_ANGLES[fin]
        for rr, aa in list(zip(r[fin], a))[::6]:
            q = (p[0] + rr * math.cos(aa), p[1] + rr * math.sin(aa))
            if q[0] < 0.15 or q[1] < 0.15 or q[0] > 9.85 or q[1] > 9.85:
                continue
            lid_err.append(world.clearance(*q)); lid_rng.append(rr)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    bins = np.arange(0.5, 5.6, 0.5)
    def binned(rn, er, bins):
        rn, er = np.array(rn), np.array(er); mu, sd, xc = [], [], []
        for lo_, hi_ in zip(bins[:-1], bins[1:]):
            m_ = (rn >= lo_) & (rn < hi_)
            if m_.sum() > 5:
                mu.append(np.mean(er[m_])); sd.append(np.percentile(er[m_], 90)); xc.append((lo_ + hi_) / 2)
        return xc, mu, sd
    xc, mu, p90 = binned(cam_rng, cam_err, bins); ax[0].plot(xc, np.array(mu) * 100, "o-", label="camera mean")
    ax[0].plot(xc, np.array(p90) * 100, "s--", label="camera 90th pct")
    xl, ml, pl = binned(lid_rng, lid_err, bins); ax[0].plot(xl, np.array(ml) * 100, "o-", label="LiDAR mean")
    ax[0].plot(xl, np.array(pl) * 100, "s--", label="LiDAR 90th pct")
    ax[0].set(title="Surface-localisation error vs range", xlabel="range [m]", ylabel="error [cm]"); ax[0].legend(fontsize=8)
    ax[1].hist(np.array(cam_err) * 100, bins=40, alpha=0.6, label="camera", density=True)
    ax[1].hist(np.array(lid_err) * 100, bins=40, alpha=0.6, label="LiDAR", density=True)
    ax[1].set(title="Error distribution", xlabel="error [cm]", ylabel="density", xlim=(0, 40)); ax[1].legend()
    # landmarks
    truth = {o["color"]: (o["cx"], o["cy"]) for o in world.objects if o["kind"] == "cyl"}
    lrows = []
    for c_, p in lm.items():
        e = float(np.hypot(*(p - np.array(truth[c_]))))
        lrows.append(dict(landmark=c_, true_x=truth[c_][0], true_y=truth[c_][1], est_x=p[0], est_y=p[1],
                          error_cm=e * 100, n_obs=len(grid.landmarks[c_])))
    ldf = pd.DataFrame(lrows); save_table(ldf.round(3), "landmark_localisation")
    ax[2].bar(ldf["landmark"], ldf["error_cm"], color=[COLORS[c_] for c_ in ldf["landmark"]])
    ax[2].set(title="Landmark localisation error (median of observations)", ylabel="error [cm]")
    fig.tight_layout(); savefig(fig, "fig05_perception_accuracy.png")
    summ = dict(cam_mean_err_cm=float(np.mean(cam_err) * 100), cam_p90_cm=float(np.percentile(cam_err, 90) * 100),
                lid_mean_err_cm=float(np.mean(lid_err) * 100), lid_p90_cm=float(np.percentile(lid_err, 90) * 100),
                survey_time_s=float(sim.t), n_cam_pts=len(cam_err), n_lid_pts=len(lid_err))
    json.dump(summ, open(os.path.join(TAB, "perception_summary.json"), "w"), indent=1)
    return df, ldf, summ


def pickle_load_traj():
    return pickle.load(open(SURVEY_PKL, "rb"))["traj"]


if __name__ == "__main__":
    df, ldf, summ = run()
    print(df.round(3).to_string()); print(ldf.round(3).to_string()); print(summ)
