"""
ORCA-VFD Research Code
Author: Carl L. Tolbert
Focus: Control-aware VFD sizing, torque-centric analysis, and reliability engineering
Context: Experimental and analytical research
Status: Non-safety-rated. Use for study, testing, and interpretation only.
License: MIT

Principle: Limits should be reached intentionally, not accidentally.
"""

import math
import csv
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Config:
    # Timing
    dt_s: float = 0.02
    total_s: float = 30 * 60

    # Speed scaling (percent)
    speed_min: float = 0.0
    speed_max: float = 100.0

    # Controller limits
    trim_limit_pct: float = 15.0
    rate_limit_pct_per_s: float = 30.0

    # Control target (controls on estimated tension)
    tension_sp: float = 50.0
    kp_trim: float = 0.25

    # True tension plant
    tension_base: float = 50.0
    tau_s: float = 1.2
    k_pull: float = 1.8
    tension_noise_units: float = 0.8

    # Proxy "current" model (sensorless input)
    proxy_i_base: float = 5.0
    k_i_tension: float = 0.020         # current units per tension unit (primary coupling)
    k_i_mismatch: float = 0.020        # current units per % mismatch (confounder)
    proxy_i_noise: float = 0.05
    proxy_i_alpha: float = 0.10        # low-pass on proxy

    # Estimator filtering (PLC-style)
    tension_est_alpha: float = 0.05

    # NEW: slow calibration drift applied to the estimated tension (units, peak)
    est_bias_drift_units: float = 3.0

    # Metrics window
    steady_start_s: float = 300.0
    steady_end_s: float = 900.0
    band_frac: float = 0.05            # ±5% of setpoint band


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rate_limit(prev: float, target: float, rate_per_s: float, dt: float) -> float:
    max_step = rate_per_s * dt
    delta = clamp(target - prev, -max_step, max_step)
    return prev + delta


def schedule_line_speed_pct(t: float) -> float:
    if t < 120:
        return 0.0
    if t < 300:
        return (t - 120) / (300 - 120) * 60.0
    if t < 900:
        return 60.0
    if t < 1500:
        base = 60.0
        if 900 <= t < 960:
            base = 66.0
        return base
    if t < 1680:
        return 60.0
    if t <= 1800:
        return 60.0 * (1.0 - (t - 1680) / (1800 - 1680))
    return 0.0


def disturbance_offset_units(t: float) -> float:
    if 1020 <= t < 1030:
        return 10.0
    if 1140 <= t < 1150:
        return -10.0
    if 1260 <= t < 1290:
        return 6.0 * math.sin(2.0 * math.pi * 0.2 * (t - 1260))
    if 1380 <= t < 1440:
        return 6.0 * ((t - 1380) / 60.0)
    return 0.0


def controller(cfg: Config, line_speed: float, tension_est: float, prev_cmd: float) -> Dict[str, float]:
    err = cfg.tension_sp - tension_est
    trim = clamp(cfg.kp_trim * err, -cfg.trim_limit_pct, cfg.trim_limit_pct)

    if line_speed < 1.0:
        cmd_target = 0.0
        trim = 0.0
    else:
        cmd_target = clamp(line_speed + trim, cfg.speed_min, cfg.speed_max)

    cmd = rate_limit(prev_cmd, cmd_target, cfg.rate_limit_pct_per_s, cfg.dt_s)
    return {"cmd": cmd, "trim": trim, "err": err}


def tension_plant(cfg: Config, tension_true: float, winder_cmd: float, line_speed: float, dist: float, rng: np.random.Generator) -> float:
    mismatch = winder_cmd - line_speed
    dtension_dt = (cfg.k_pull * mismatch) - ((tension_true - cfg.tension_base) / cfg.tau_s)
    noise = rng.normal(0.0, cfg.tension_noise_units)
    return tension_true + dtension_dt * cfg.dt_s + dist * 0.2 + noise


def proxy_current_update(
    cfg: Config,
    proxy_i_filt: float,
    tension_true: float,
    winder_cmd: float,
    line_speed: float,
    rng: np.random.Generator
) -> float:
    mismatch = abs(winder_cmd - line_speed)

    raw = (
        cfg.proxy_i_base
        + cfg.k_i_tension * (tension_true - cfg.tension_base)
        + cfg.k_i_mismatch * mismatch
        + rng.normal(0.0, cfg.proxy_i_noise)
    )

    a = cfg.proxy_i_alpha
    return (1.0 - a) * proxy_i_filt + a * raw


def estimate_tension(cfg: Config, proxy_i_filt: float, tension_est_filt: float) -> float:
    # Inverse map: ignores mismatch contribution intentionally (realistic penalty)
    tension_est_raw = cfg.tension_base + (proxy_i_filt - cfg.proxy_i_base) / max(cfg.k_i_tension, 1e-9)

    # Clamp to plausible range
    tension_est_raw = clamp(tension_est_raw, 0.0, 2.0 * cfg.tension_sp)

    a = cfg.tension_est_alpha
    return (1.0 - a) * tension_est_filt + a * tension_est_raw


def run_sim(cfg: Config) -> List[Dict[str, float]]:
    rng = np.random.default_rng(42)
    steps = int(cfg.total_s / cfg.dt_s)

    rows: List[Dict[str, float]] = []

    winder_cmd = 0.0
    tension_true = cfg.tension_base

    proxy_i_filt = cfg.proxy_i_base
    tension_est_filt = cfg.tension_base

    base_k_pull = cfg.k_pull

    for k in range(steps):
        t = k * cfg.dt_s
        line = schedule_line_speed_pct(t)
        dist = disturbance_offset_units(t)

        # Small-roll sensitivity segment
        if 1500 <= t < 1680:
            cfg.k_pull = base_k_pull * 1.5
        else:
            cfg.k_pull = base_k_pull

        # 1) Plant evolves based on previous command
        tension_true = tension_plant(cfg, tension_true, winder_cmd, line, dist, rng)

        # 2) Proxy current derived from true tension + small mismatch effect
        proxy_i_filt = proxy_current_update(cfg, proxy_i_filt, tension_true, winder_cmd, line, rng)

        # 3) Tension estimate from proxy
        tension_est_filt = estimate_tension(cfg, proxy_i_filt, tension_est_filt)

        # NEW: slow drift/bias applied to estimate (calibration drift over run)
        drift = cfg.est_bias_drift_units * math.sin(2.0 * math.pi * (t / cfg.total_s))
        tension_est_used = tension_est_filt + drift

        # 4) Control generates next command using drifted estimate
        ctl = controller(cfg, line, tension_est_used, winder_cmd)
        winder_cmd = ctl["cmd"]

        rows.append({
            "t_s": t,
            "line_speed_pct": line,
            "winder_cmd_pct": winder_cmd,
            "trim_pct": ctl["trim"],
            "proxy_current_filt": proxy_i_filt,
            "tension_est": tension_est_filt,           # filtered estimate before drift
            "tension_est_used": tension_est_used,      # estimate used by controller (with drift)
            "tension_true": tension_true,
            "tension_err_est": ctl["err"],             # setpoint - (estimate_used)
            "tension_est_minus_true": (tension_est_used - tension_true),
        })

    return rows


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def pct_time(condition: np.ndarray) -> float:
    return float(np.mean(condition) * 100.0)


def compute_metrics(cfg: Config, rows: List[Dict[str, float]]) -> Dict[str, float]:
    t = np.array([r["t_s"] for r in rows])

    tension_est_used = np.array([r["tension_est_used"] for r in rows])
    tension_true = np.array([r["tension_true"] for r in rows])
    err_est_used = np.array([r["tension_err_est"] for r in rows])
    trim = np.array([r["trim_pct"] for r in rows])

    steady_mask = (t >= cfg.steady_start_s) & (t < cfg.steady_end_s)

    band = cfg.band_frac * cfg.tension_sp

    # Control performance (on estimate used by controller)
    err_rms_est = rms(err_est_used[steady_mask])
    within_band_est = pct_time(np.abs(tension_est_used[steady_mask] - cfg.tension_sp) <= band)

    # True tension performance (what physically matters)
    err_true = (cfg.tension_sp - tension_true)
    err_rms_true = rms(err_true[steady_mask])
    within_band_true = pct_time(np.abs(tension_true[steady_mask] - cfg.tension_sp) <= band)

    # Estimator quality vs true (using estimate used for control)
    est_minus_true_rms = rms((tension_est_used[steady_mask] - tension_true[steady_mask]))

    # Actuator saturation
    trim_sat = pct_time(np.abs(trim) >= (cfg.trim_limit_pct - 1e-9))

    return {
        "band_units": band,
        "steady_est_error_rms": err_rms_est,
        "time_within_band_est_pct": within_band_est,
        "steady_true_error_rms": err_rms_true,
        "time_within_band_true_pct": within_band_true,
        "trim_saturation_time_pct": trim_sat,
        "est_minus_true_rms": est_minus_true_rms,
    }


def save_figures(cfg: Config, rows: List[Dict[str, float]]) -> None:
    t = np.array([r["t_s"] for r in rows]) / 60.0
    line = np.array([r["line_speed_pct"] for r in rows])
    cmd = np.array([r["winder_cmd_pct"] for r in rows])
    trim = np.array([r["trim_pct"] for r in rows])

    tension_true = np.array([r["tension_true"] for r in rows])
    tension_est_used = np.array([r["tension_est_used"] for r in rows])

    # Figure 1: Speed
    plt.figure()
    plt.plot(t, line, label="Line speed (%)")
    plt.plot(t, cmd, label="Winder cmd (%)")
    plt.xlabel("Time (min)")
    plt.ylabel("Speed (%)")
    plt.legend()
    plt.title("Speed profile and winder command (sensorless mode)")
    plt.savefig("sensorless_figure_1_speed.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Trim
    plt.figure()
    plt.plot(t, trim)
    plt.xlabel("Time (min)")
    plt.ylabel("Trim (%)")
    plt.title("Speed trim command (sensorless mode)")
    plt.savefig("sensorless_figure_2_trim.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3: True vs estimate used for control
    plt.figure()
    plt.plot(t, tension_true, label="True tension")
    plt.plot(t, tension_est_used, label="Estimated tension (used)")
    plt.axhline(cfg.tension_sp, linestyle="--", label="Setpoint")
    plt.xlabel("Time (min)")
    plt.ylabel("Tension (units)")
    plt.legend()
    plt.title("Sensorless estimated tension (used) vs true tension")
    plt.savefig("sensorless_figure_3_tension.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    cfg = Config()

    rows = run_sim(cfg)
    write_csv("sim_sensorless.csv", rows)
    metrics = compute_metrics(cfg, rows)
    save_figures(cfg, rows)

    print("Wrote sim_sensorless.csv with", len(rows), "rows")
    print("Saved sensorless_figure_1_speed.png, sensorless_figure_2_trim.png, sensorless_figure_3_tension.png")
    print("=== Sensorless Metrics ===")
    print(f"Steady estimated tension error RMS (units): {metrics['steady_est_error_rms']:.3f}")
    print(f"Steady TRUE tension error RMS (units): {metrics['steady_true_error_rms']:.3f}")
    print(f"Trim saturation time (%): {metrics['trim_saturation_time_pct']:.2f}")
    print(
        f"Time within ±{metrics['band_units']:.2f} units (±{int(cfg.band_frac*100)}% of setpoint) on EST (%): "
        f"{metrics['time_within_band_est_pct']:.2f}"
    )
    print(
        f"Time within ±{metrics['band_units']:.2f} units (±{int(cfg.band_frac*100)}% of setpoint) on TRUE (%): "
        f"{metrics['time_within_band_true_pct']:.2f}"
    )
    print(f"Estimator RMS (estimated-used minus true) (units): {metrics['est_minus_true_rms']:.3f}")
