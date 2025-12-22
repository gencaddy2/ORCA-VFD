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

    # Load cell setpoint and controller gain
    tension_sp: float = 50.0
    kp_trim: float = 0.40

    # Tension plant (baseline + dynamics)
    tension_base: float = 50.0     # equilibrium tension at zero mismatch
    tau_s: float = 1.2
    k_pull: float = 1.8            # sensitivity of tension to speed mismatch
    noise_units: float = 0.8

    # Measurement filtering (simulates transmitter/PLC filtering)
    tension_filter_alpha: float = 0.05   # 0.02–0.15 typical

    # Metrics window
    steady_start_s: float = 300.0
    steady_end_s: float = 900.0
    band_frac: float = 0.05              # ±5% of setpoint band


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rate_limit(prev: float, target: float, rate_per_s: float, dt: float) -> float:
    max_step = rate_per_s * dt
    delta = clamp(target - prev, -max_step, max_step)
    return prev + delta


def schedule_line_speed_pct(t: float) -> float:
    # Phase 0: 0-120s idle
    if t < 120:
        return 0.0
    # Phase 1: 120-300s ramp to 60%
    if t < 300:
        return (t - 120) / (300 - 120) * 60.0
    # Phase 2: 300-900s steady 60%
    if t < 900:
        return 60.0
    # Phase 3: 900-1500s disturbances with base 60%
    if t < 1500:
        base = 60.0
        # 15:00-16:00 speed step +10%
        if 900 <= t < 960:
            base = 66.0
        return base
    # Phase 4: 1500-1680s small-roll sensitivity segment
    if t < 1680:
        return 60.0
    # Phase 5: 1680-1800s ramp down
    if t <= 1800:
        return 60.0 * (1.0 - (t - 1680) / (1800 - 1680))
    return 0.0


def disturbance_offset_units(t: float) -> float:
    # Disturbances bias tension directly (drag changes, etc.)
    if 1020 <= t < 1030:
        return 10.0   # drag increase
    if 1140 <= t < 1150:
        return -10.0  # drag decrease / slack-like
    if 1260 <= t < 1290:
        return 6.0 * math.sin(2.0 * math.pi * 0.2 * (t - 1260))
    if 1380 <= t < 1440:
        return 6.0 * ((t - 1380) / 60.0)
    return 0.0


def loadcell_controller(cfg: Config, line_speed: float, tension_meas: float, prev_cmd: float) -> Dict[str, float]:
    err = cfg.tension_sp - tension_meas
    trim = clamp(cfg.kp_trim * err, -cfg.trim_limit_pct, cfg.trim_limit_pct)

    # Startup gating for cleaner plots/behavior
    if line_speed < 1.0:
        cmd_target = 0.0
        trim = 0.0
    else:
        cmd_target = clamp(line_speed + trim, cfg.speed_min, cfg.speed_max)

    cmd = rate_limit(prev_cmd, cmd_target, cfg.rate_limit_pct_per_s, cfg.dt_s)
    return {"cmd": cmd, "trim": trim, "err": err}


def tension_plant(cfg: Config, tension: float, winder_cmd: float, line_speed: float, dist: float, rng: np.random.Generator) -> float:
    mismatch = winder_cmd - line_speed

    # Relax toward baseline equilibrium (not toward zero)
    dtension_dt = (cfg.k_pull * mismatch) - ((tension - cfg.tension_base) / cfg.tau_s)

    noise = rng.normal(0.0, cfg.noise_units)
    tension_next = tension + dtension_dt * cfg.dt_s + dist * 0.2 + noise
    return tension_next


def run_sim(cfg: Config) -> List[Dict[str, float]]:
    rng = np.random.default_rng(42)
    steps = int(cfg.total_s / cfg.dt_s)

    rows: List[Dict[str, float]] = []
    winder_cmd = 0.0

    tension_raw = cfg.tension_base
    tension_filt = tension_raw

    base_k_pull = cfg.k_pull

    for k in range(steps):
        t = k * cfg.dt_s
        line = schedule_line_speed_pct(t)
        dist = disturbance_offset_units(t)

        # Small-roll sensitivity segment: increase plant gain
        if 1500 <= t < 1680:
            cfg.k_pull = base_k_pull * 1.5
        else:
            cfg.k_pull = base_k_pull

        # Control based on filtered measurement (realistic)
        ctl = loadcell_controller(cfg, line, tension_filt, winder_cmd)
        winder_cmd = ctl["cmd"]

        # Plant update gives raw measurement
        tension_raw = tension_plant(cfg, tension_raw, winder_cmd, line, dist, rng)

        # Low-pass filter (transmitter/PLC filter behavior)
        a = cfg.tension_filter_alpha
        tension_filt = (1.0 - a) * tension_filt + a * tension_raw

        rows.append({
            "t_s": t,
            "line_speed_pct": line,
            "winder_cmd_pct": winder_cmd,
            "trim_pct": ctl["trim"],
            "tension_meas_raw": tension_raw,
            "tension_meas": tension_filt,
            "tension_err": ctl["err"],
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
    tension = np.array([r["tension_meas"] for r in rows])   # filtered
    err = np.array([r["tension_err"] for r in rows])
    trim = np.array([r["trim_pct"] for r in rows])

    steady_mask = (t >= cfg.steady_start_s) & (t < cfg.steady_end_s)
    err_rms = rms(err[steady_mask])

    trim_sat = pct_time(np.abs(trim) >= (cfg.trim_limit_pct - 1e-9))

    band = cfg.band_frac * cfg.tension_sp
    within_band = pct_time(np.abs(tension[steady_mask] - cfg.tension_sp) <= band)

    return {
        "steady_tension_error_rms": err_rms,
        "trim_saturation_time_pct": trim_sat,
        "time_within_band_pct": within_band,
        "band_units": band
    }


def save_figures(cfg: Config, rows: List[Dict[str, float]]) -> None:
    t = np.array([r["t_s"] for r in rows]) / 60.0
    line = np.array([r["line_speed_pct"] for r in rows])
    cmd = np.array([r["winder_cmd_pct"] for r in rows])
    trim = np.array([r["trim_pct"] for r in rows])
    tension_raw = np.array([r["tension_meas_raw"] for r in rows])
    tension_filt = np.array([r["tension_meas"] for r in rows])

    # Figure 1
    plt.figure()
    plt.plot(t, line, label="Line speed (%)")
    plt.plot(t, cmd, label="Winder cmd (%)")
    plt.xlabel("Time (min)")
    plt.ylabel("Speed (%)")
    plt.legend()
    plt.title("Speed profile and winder command (load cell mode)")
    plt.savefig("loadcell_figure_1_speed.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2
    plt.figure()
    plt.plot(t, trim)
    plt.xlabel("Time (min)")
    plt.ylabel("Trim (%)")
    plt.title("Speed trim command (load cell mode)")
    plt.savefig("loadcell_figure_2_trim.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3 (raw + filtered)
    plt.figure()
    plt.plot(t, tension_raw, label="Raw tension")
    plt.plot(t, tension_filt, label="Filtered tension")
    plt.axhline(cfg.tension_sp, linestyle="--", label="Setpoint")
    plt.xlabel("Time (min)")
    plt.ylabel("Tension (units)")
    plt.legend()
    plt.title("Load cell tension response (raw vs filtered)")
    plt.savefig("loadcell_figure_3_tension.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    cfg = Config()

    rows = run_sim(cfg)
    write_csv("sim_loadcell.csv", rows)
    metrics = compute_metrics(cfg, rows)
    save_figures(cfg, rows)

    print("Wrote sim_loadcell.csv with", len(rows), "rows")
    print("Saved loadcell_figure_1_speed.png, loadcell_figure_2_trim.png, loadcell_figure_3_tension.png")
    print("=== Load Cell Metrics ===")
    print(f"Steady tension error RMS (units): {metrics['steady_tension_error_rms']:.3f}")
    print(f"Trim saturation time (%): {metrics['trim_saturation_time_pct']:.2f}")
    print(
        f"Time within ±{metrics['band_units']:.2f} units (±{int(cfg.band_frac*100)}% of setpoint) (%): "
        f"{metrics['time_within_band_pct']:.2f}"
    )
