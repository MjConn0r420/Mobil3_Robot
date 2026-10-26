from common import *
from navlib import config as C
from navlib.control import PID, wrap, PathTracker
from navlib.sim import RobotSim
from navlib.planning import resample

TD = C.DELAY_STEPS * C.DT + C.DT / 2          # delay + zero-order-hold
TAU = C.TAU_MOTOR
W_BIAS = 0.08                                 # constant angular-rate disturbance used in step tests [rad/s]


# ---------------- linear (frequency-domain) analysis ----------------
def loop_tf(gains, w):
    kp, ki, kd = gains
    s = 1j * w
    Cc = kp + ki / s + kd * s / (0.05 * s + 1)
    G = np.exp(-s * TD) / (s * (TAU * s + 1))
    return Cc * G


def margins(gains):
    w = np.logspace(-2, 2.3, 40000)
    L = loop_tf(gains, w)
    mag, ph = np.abs(L), np.unwrap(np.angle(L))
    ic = np.nonzero((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    pm, wc = np.nan, np.nan
    if len(ic):
        i = ic[-1]; wc = w[i]; pm = 180 + np.degrees(ph[i])
    ip = np.nonzero((ph[:-1] > -np.pi) & (ph[1:] <= -np.pi))[0]
    gm, wp = np.inf, np.nan
    if len(ip):
        i = ip[0]; wp = w[i]; gm = 1 / mag[i]
    return dict(PM_deg=pm, GM_db=20 * np.log10(gm) if np.isfinite(gm) else np.inf, wc=wc, wp=wp)


def ultimate():
    w = np.logspace(-1, 2, 200000)
    G = np.exp(-1j * w * TD) / (1j * w * (TAU * 1j * w + 1))
    ph = np.unwrap(np.angle(G))
    i = np.nonzero(ph <= -np.pi)[0][0]
    return 1 / abs(G[i]), 2 * np.pi / w[i]


# ---------------- time-domain tests ----------------
def heading_step(gains, aw=True, T=8.0, seed=3, bias=W_BIAS, ref_deg=90):
    rng = np.random.default_rng(seed)
    sim = RobotSim((0, 0, 0.0), None, rng, w_bias=bias)
    pid = PID(*gains, anti_windup=aw)
    ref = math.radians(ref_deg)
    t, th, wc = [], [], []
    for k in range(int(T / C.DT)):
        e = wrap(ref - sim.measure()[2])
        w = pid.update(e, C.DT)
        sim.step(0.0, w)
        t.append(sim.t); th.append(sim.th); wc.append(w)
    return np.array(t), np.array(th), np.array(wc), ref


def lateral_recovery(gains, aw=True, T=14.0, seed=4, offset=-0.5, v=0.4):
    rng = np.random.default_rng(seed)
    path = resample(np.array([[0.0, 0.0], [10.0, 0.0]]), 0.05)
    sim = RobotSim((0.0, offset, 0.0), None, rng, w_bias=0.03, eps=0.01)
    trk = PathTracker(path, gains, v_max=v, lookahead=0.4, anti_windup=aw)
    t, y = [], []
    for k in range(int(T / C.DT)):
        vv, w, done, e = trk.command(sim.measure(), C.DT)
        sim.step(vv, w)
        t.append(sim.t); y.append(sim.y)
    return np.array(t), np.array(y), offset


def evaluate(name, gains, aw=True):
    t, th, wc, ref = heading_step(gains, aw)
    mh = step_metrics(t, th, ref, 0.0)
    tail = t >= t[-1] - 2.0
    jitter = float(np.std(wc[tail]))
    t2, y, y0 = lateral_recovery(gains, aw)
    ml = step_metrics(t2, y, 0.0, y0)
    mg = margins(gains)
    return dict(set=name, Kp=gains[0], Ki=gains[1], Kd=gains[2], anti_windup=aw,
                head_OS_pct=mh["overshoot_pct"], head_rise_s=mh["rise_s"], head_settle_s=mh["settle_s"],
                head_SSE_deg=math.degrees(mh["sse"]), w_jitter=jitter,
                lat_OS_pct=ml["overshoot_pct"], lat_rise_s=ml["rise_s"], lat_settle_s=ml["settle_s"],
                lat_SSE_cm=ml["sse"] * 100, PM_deg=mg["PM_deg"], GM_db=mg["GM_db"])


def tune():
    """Grid search over (Kp,Ki,Kd). Cost = ITAE + overshoot + settling on BOTH the heading step and the
    lateral-offset recovery, subject to a linear phase-margin constraint PM >= 45 deg."""
    best, rows = None, []
    for kp in [1.5, 2, 3, 4, 5, 6, 8]:
        for ki in [0, 0.1, 0.3, 0.6, 1.0, 2.0]:
            for kd in [0, 0.1, 0.2, 0.4, 0.8]:
                g = (kp, ki, kd)
                mg = margins(g)
                if not (mg["PM_deg"] >= 45):
                    continue
                t, th, wc, ref = heading_step(g)
                mh = step_metrics(t, th, ref, 0.0)
                t2, y, y0 = lateral_recovery(g)
                ml = step_metrics(t2, y, 0.0, y0)
                def cost(m, T):
                    st = m["settle_s"] if not np.isnan(m["settle_s"]) else T
                    return m["itae"] * 10 + m["overshoot_pct"] / 10 + 0.3 * st + 20 * abs(m["sse"])
                # penalise derivative noise amplification
                jit = float(np.std(wc[t >= t[-1] - 2.0]))
                J = cost(mh, 8) + cost(ml, 14) + 5 * jit
                rows.append(dict(Kp=kp, Ki=ki, Kd=kd, J=J, PM=mg["PM_deg"]))
                if best is None or J < best[0]:
                    best = (J, g)
    return best[1], pd.DataFrame(rows).sort_values("J")


def run():
    Ku, Pu = ultimate()
    zn = (0.6 * Ku, 1.2 * Ku / Pu, 0.075 * Ku * Pu)
    tuned, search = tune()
    print("Ku, Pu =", Ku, Pu, " ZN =", zn, " tuned =", tuned)
    sets = [
        ("A  P low", (1.0, 0, 0), True),
        ("B  P high (0.8 Ku)", (0.8 * Ku, 0, 0), True),
        ("C  PI, no anti-windup", (3.0, 2.0, 0.0), False),
        ("D  PI, anti-windup", (3.0, 2.0, 0.0), True),
        ("E  Ziegler-Nichols PID", zn, True),
        ("F  PID high Kd", (tuned[0], tuned[1], 1.5), True),
        ("G  PID tuned (grid search)", tuned, True),
    ]
    rows = [evaluate(n, g, aw) for n, g, aw in sets]
    df = pd.DataFrame(rows)
    save_table(df.round(3), "pid_tuning_metrics")
    save_table(search.head(10).round(3), "pid_grid_search_top10")
    json.dump({"Kp": tuned[0], "Ki": tuned[1], "Kd": tuned[2], "Ku": Ku, "Pu": Pu,
               "zn": list(zn)}, open(os.path.join(ROOT, "navlib", "tuned_gains.json"), "w"), indent=1)

    # ---- figure: heading step responses
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    pal = sns.color_palette("tab10", len(sets))
    for (n, g, aw), c in zip(sets, pal):
        t, th, wc, ref = heading_step(g, aw)
        ax[0, 0].plot(t, np.degrees(th), color=c, label=n)
        ax[1, 0].plot(t, wc, color=c, lw=1)
        t2, y, y0 = lateral_recovery(g, aw)
        ax[0, 1].plot(t2, y, color=c, label=n)
        ax[1, 1].plot(t2, np.gradient(y, t2), color=c, lw=0.8)
    ax[0, 0].axhline(90, color="k", ls="--", lw=0.8)
    ax[0, 0].axhspan(90 * 0.98, 90 * 1.02, color="k", alpha=0.08)
    ax[0, 0].set(title="Heading step 0 -> 90 deg (with 0.08 rad/s disturbance)", ylabel="heading [deg]", xlabel="t [s]")
    ax[0, 0].legend(fontsize=7.5, loc="lower right")
    ax[1, 0].set(title="Angular-rate command (saturates at +/-2 rad/s)", ylabel="w_cmd [rad/s]", xlabel="t [s]")
    ax[0, 1].axhline(0, color="k", ls="--", lw=0.8)
    ax[0, 1].set(title="Lateral-offset recovery: cross-track y(t) (start 0.5 m off the line)", ylabel="y [m]", xlabel="t [s]")
    ax[1, 1].set(title="Lateral velocity", ylabel="dy/dt [m/s]", xlabel="t [s]")
    fig.tight_layout(); savefig(fig, "fig09_pid_step_responses.png")

    # ---- figure: Bode + margins
    w = np.logspace(-1, 2, 2000)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for (n, g, aw), c in zip(sets, pal):
        L = loop_tf(g, w)
        ax[0].semilogx(w, 20 * np.log10(np.abs(L)), color=c, label=n)
        ax[1].semilogx(w, np.degrees(np.unwrap(np.angle(L))), color=c)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set(title="Open-loop magnitude", xlabel="w [rad/s]", ylabel="dB")
    ax[1].axhline(-180, color="k", lw=0.8); ax[1].set(title="Open-loop phase", xlabel="w [rad/s]", ylabel="deg", ylim=(-340, -60))
    ax[0].legend(fontsize=7)
    kps = np.linspace(0.5, 1.6 * Ku, 120)
    pms = [margins((k, 0, 0))["PM_deg"] for k in kps]
    ax[2].plot(kps, pms, color="C0", label="Phase margin (P-only)")
    ax[2].axhline(45, color="r", ls="--", lw=0.8); ax[2].axvline(Ku, color="k", ls=":", lw=1)
    ax[2].text(Ku, 5, f" Ku={Ku:.1f}", fontsize=8)
    ax[2].set(title="Stability margin vs Kp (plant 1/[s(tau s+1)] + 80 ms delay)", xlabel="Kp", ylabel="PM [deg]")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); savefig(fig, "fig11_stability_bode.png")
    return tuned, df


if __name__ == "__main__":
    tuned, df = run()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df.round(3).to_string())
