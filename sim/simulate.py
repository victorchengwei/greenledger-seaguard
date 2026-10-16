"""SIMULATED data + weather augmentation for demos. Label everything as simulated."""
import sys

import cv2
import numpy as np
import pandas as pd


def make_telemetry(hours=24, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-10-01", periods=hours * 6, freq="10min")
    n = len(t)
    df = pd.DataFrame({
        "time": t,
        "fuel_lph": 420 + rng.normal(0, 8, n),
        "engine_temp_c": 82 + rng.normal(0, 0.8, n),
        "vibration_mm_s": 2.0 + rng.normal(0, 0.15, n),
    })
    df.loc[n - 30:, "vibration_mm_s"] += np.linspace(0, 2.5, 30)  # injected fault
    df["simulated"] = True
    return df


def maintenance_alerts(df, limit=3.5):
    bad = df[df["vibration_mm_s"] > limit]
    return [] if bad.empty else [f"Vibration above {limit} mm/s since {bad['time'].iloc[0]:%H:%M} (SIMULATED)"]


def make_ais():
    """SIMULATED AIS picture off Kaohsiung (own ship + 4 targets) for the COLREG demo."""
    own = {"lat": 22.600, "lon": 120.250, "sog": 12, "cog": 90}
    targets = [
        {"mmsi": "SIM001", "class": "Cargo/Container", "status": "underway",
         "lat": 22.604, "lon": 120.290, "sog": 10, "cog": 270},
        {"mmsi": "SIM002", "class": "Fishing", "status": "engaged_in_fishing",
         "lat": 22.620, "lon": 120.300, "sog": 4, "cog": 200},
        {"mmsi": "SIM003", "class": "Tanker", "status": "underway",
         "lat": 22.560, "lon": 120.330, "sog": 8, "cog": 0},
        {"mmsi": "SIM004", "class": "Tug", "status": "towing",
         "lat": 22.575, "lon": 120.290, "sog": 6, "cog": 60},
    ]
    return own, targets


def augment(frame, fog=0.5, rain=True, seed=None):
    rng = np.random.default_rng(seed)
    out = cv2.GaussianBlur(frame, (0, 0), 1 + 2 * fog)
    haze = np.full_like(out, 205)
    out = cv2.addWeighted(out, 1 - fog * 0.6, haze, fog * 0.6, 0)
    if rain:
        layer = np.zeros(out.shape[:2], np.uint8)
        h, w = layer.shape
        for _ in range(int(w * h / 900)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.line(layer, (x, y), (x - 4, y + 14), 255, 1)
        out = cv2.addWeighted(out, 1, cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR), 0.35, 0)
    return out


if __name__ == "__main__":  # python sim/simulate.py in.mp4 out.mp4 0.5
    src, dst = sys.argv[1], sys.argv[2]
    fog = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(3)), int(cap.get(4)))
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vw.write(augment(f, fog))
    vw.release()
