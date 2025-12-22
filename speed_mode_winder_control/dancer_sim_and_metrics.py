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

    # Controller
    dancer_mid_pct: float = 50.0
    kp_trim: float = 0.40
    trim_limit_pct: float = 15.0
    rate_limit_pct_per_s: float = 30.0

    # Plant
    tau_s: float = 1.5
    k_elastic: float = 0.020
    noise_pct: float = 0.15

    # Metrics
    steady_start_s: float = 300.0   # 5 min
    steady_end_s: float = 900.0     # 15 min
    band_pct_points: float = 2.0    # ±2%-points band


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


def disturbance_offset_pct_points(t: float) -> float:
    # Disturbances that bias dancer position
    # 17:00-18:00 slack-like bias
    if 1020 <= t < 1030:
        return 6.0
    # 19:00-20:00 tight-like bias
    if 1140 <= t < 1150:
        return -6.0
    # 21:00-22:00 oscillation burst for 30s
    if 1260 <= t < 1290:
        return 3.0 * math.sin(2.0 * math.pi * 0.2 * (t - 1260))
    # 23:00-24:00 drift over 60s
    if 1380 <= t < 1440:
        return 3.0 * ((t - 1380) / 60.0)
    return 0.0


def dancer_controller(cfg: Config, line_speed: float, dancer_pos: float, prev_cmd: float) -> Dict[str, float]:
    # Error sign convention:
    # dancer above mid implies slack, pull more, increase speed
    err = cfg.dancer_mid_pct - dancer_pos
    trim = clamp(cfg.kp_trim * err, -cfg.trim_limit_pct, cfg.trim_limit_pct)

    # Optional startup gating for cleaner behavior at zero speed
    if line_speed < 1.0:
        cmd_target = 0.0
        trim = 0.0
        err = cfg.dancer_mid_pct - dancer_pos
    else:
        cmd_target = clamp(line_speed + trim, cfg.speed_min, cfg.speed_max)

    cmd = rate_limit(prev_cmd, cmd_target, cfg.rate_limit_pct_per_s, cfg.dt_s)
    return {"cmd": cmd, "trim": trim, "err": err}


def dancer_plant(cfg: Config, dancer_pos: float, winder_cmd: float, line_speed: float, dist: float, rng: np.random.Generator) -> float:
    mismatch = winder_cmd - line_speed
    # First-order dynamics + relaxation toward mid
    dpos_dt = (cfg.k_elastic * mismatch) - ((dancer_pos - cfg.dancer_mid_pct) / cfg.tau_s)

    noise = rng.normal(0.0, cfg.noise_pct)
    dancer_next = dancer_pos + dpos_dt * cfg.dt_s + dist * cfg.dt_s + noise
    return clamp(dancer_next, 0.0, 100.0)


def run_sim(cfg: Config) -> List[Dict[str, float]]:
    rng = np.random.default_rng(42)
    steps = int(cfg.total_s / cfg.dt_s)

    rows: List[Dict[str, float]] = []
    winder_cmd = 0.0
    dancer_pos = cfg.dancer_mid_pct

    base_k_elastic = cfg.k_elastic

    for k in range(steps):
        t = k * cfg.dt_s
        line = schedule_line_speed_pct(t)
        dist = disturbance_offset_pct_points(t)

        # Small-roll sensitivity segment: increase plant gain
        if 1500 <= t < 1680:
            cfg.k_elastic = base_k_elastic * 1.5
        else:
            cfg.k_elastic = base_k_elastic

        ctl = dancer_controller(cfg, line, dancer_pos, winder_cmd)
        winder_cmd = ctl["cmd"]
        dancer_pos = dancer_plant(cfg, dancer_pos, winder_cmd, line, dist, rng)

        rows.append({
            "t_s": t,
            "line_speed_pct": line,
            "winder_cmd_pct": winder_cmd,
            "trim_pct": ctl["trim"],
            "dancer_pos_pct": dancer_pos,
            "dancer_err": ctl["err"],
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
    dancer = np.array([r["dancer_pos_pct"] for r in rows])
    err = np.array([r["dancer_err"] for r in rows])
    trim = np.array([r["trim_pct"] for r in rows])

    steady_mask = (t >= cfg.steady_start_s) & (t < cfg.steady_end_s)

    err_rms = rms(err[steady_mask])
    pos_rms = rms((dancer[steady_mask] - cfg.dancer_mid_pct))
    trim_sat = pct_time(np.abs(trim) >= (cfg.trim_limit_pct - 1e-9))
    within_band = pct_time(np.abs(dancer[steady_mask] - cfg.dancer_mid_pct) <= cfg.band_pct_points)

    return {
        "steady_err_rms_pct_points": err_rms,
        "steady_pos_rms_pct_points": pos_rms,
        "trim_saturation_time_pct": trim_sat,
        f"time_within_pm_{cfg.band_pct_points:.0f}_pct_points_pct": within_band
    }


def save_figures(cfg: Config, rows: List[Dict[str, float]]) -> None:
    t = np.array([r["t_s"] for r in rows]) / 60.0
    line = np.array([r["line_speed_pct"] for r in rows])
    cmd = np.array([r["winder_cmd_pct"] for r in rows])
    trim = np.array([r["trim_pct"] for r in rows])
    dancer = np.array([r["dancer_pos_pct"] for r in rows])

    # Figure 1
    plt.figure()
    plt.plot(t, line, label="Line speed (%)")
    plt.plot(t, cmd, label="Winder cmd (%)")
    plt.xlabel("Time (min)")
    plt.ylabel("Speed (%)")
    plt.legend()
    plt.title("Speed profile and winder command (dancer mode)")
    plt.savefig("dancer_figure_1_speed.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2
    plt.figure()
    plt.plot(t, trim)
    plt.xlabel("Time (min)")
    plt.ylabel("Trim (%)")
    plt.title("Speed trim command (dancer mode)")
    plt.savefig("dancer_figure_2_trim.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3
    plt.figure()
    plt.plot(t, dancer)
    plt.axhline(cfg.dancer_mid_pct, linestyle="--")
    plt.xlabel("Time (min)")
    plt.ylabel("Dancer position (%)")
    plt.title("Dancer position response")
    plt.savefig("dancer_figure_3_dancer.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    cfg = Config()

    rows = run_sim(cfg)

    write_csv("sim_dancer.csv", rows)
    metrics = compute_metrics(cfg, rows)
    save_figures(cfg, rows)

    print("Wrote sim_dancer.csv with", len(rows), "rows")
    print("Saved dancer_figure_1_speed.png, dancer_figure_2_trim.png, dancer_figure_3_dancer.png")
    print("=== Dancer Metrics ===")
    print(f"Steady dancer error RMS (%-points): {metrics['steady_err_rms_pct_points']:.3f}")
    print(f"Steady dancer position RMS (%-points): {metrics['steady_pos_rms_pct_points']:.3f}")
    print(f"Trim saturation time (%): {metrics['trim_saturation_time_pct']:.2f}")
    key = f"time_within_pm_{cfg.band_pct_points:.0f}_pct_points_pct"
    print(f"Time within ±{cfg.band_pct_points:.0f}%-points of mid (%): {metrics[key]:.2f}")
