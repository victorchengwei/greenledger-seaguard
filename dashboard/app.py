"""Streamlit dashboard: detection collage + vessel types + waterway map + AI alert + fuel estimate."""
import base64
import os
import sys

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.risk import assess, explain, fuel_impact                     # noqa: E402
from agent.vessel_type import classify_crops                             # noqa: E402
from sim.simulate import make_ais, make_telemetry, maintenance_alerts   # noqa: E402

st.set_page_config(page_title="GreenLedger", layout="wide")
st.title("GreenLedger — Agentic vessel collision avoidance")
api = st.sidebar.text_input("Detector API URL", os.getenv("DETECTOR_URL", "http://localhost:8080"))
weather = {"wind_kn": st.sidebar.slider("Wind (kn)", 0, 50, 18),
           "wave_height_m": st.sidebar.slider("Wave height (m)", 0.0, 6.0, 1.5),
           "visibility_nm": st.sidebar.slider("Visibility (nm)", 0.1, 10.0, 1.0)}
conf_thr = st.sidebar.slider("Detection confidence threshold", 0.05, 0.50, 0.20, 0.05,
                             help="Lower = fewer missed vessels, more false alarms. "
                                  "At sea, a false alarm costs less than a missed vessel.")
st.sidebar.caption("Gemini key set: " + ("yes" if os.getenv("GEMINI_API_KEY") else "NO (fallback text only)"))
st.sidebar.caption("Own ship assumed: power-driven vessel underway. Decision support only; "
                   "it never replaces a proper lookout (Rule 5) or the OOW's judgement.")


def colreg_card(title, d):
    """Four-line COLREG summary: type / status / responsibility / action."""
    with st.container(border=True):
        st.markdown(f"**{title}**  \n"
                    f"🚢 **船種 Vessel type:** {d.get('type') or d.get('class', '')}  \n"
                    f"⚓ **COLREG 狀態:** {d.get('colreg_status', '')}  \n"
                    f"⚠️ **避讓責任 Responsibility:** {d.get('give_way', '')}  \n"
                    f"➡️ **建議動作 Action:** {d.get('action', '')}")


def detect_frame(api_url, jpg_bytes, conf, attempts=2):
    """POST one JPEG to the detector. Retries once, because a scaled-to-zero
    Cloud Run service can time out or return 5xx while it cold-starts."""
    for attempt in range(attempts):
        try:
            resp = requests.post(f"{api_url}/detect", params={"conf": conf},
                                 files={"file": ("f.jpg", jpg_bytes, "image/jpeg")}, timeout=90)
            resp.raise_for_status()
            out = resp.json()
            out["detections"], out["annotated_jpg_b64"]   # fail fast on a malformed reply
            return out
        except requests.RequestException:
            if attempt == attempts - 1:
                raise


up = st.file_uploader("Upload a camera image or video (rain/fog clip)", type=["jpg", "jpeg", "png", "mp4"])
if up is not None:
    raw = up.read()
    sig = (up.name, len(raw), conf_thr)
    if st.session_state.get("sig") != sig:          # process each new file once
        frames = []
        if up.name.lower().endswith("mp4"):
            with open("tmp_in.mp4", "wb") as f:
                f.write(raw)
            cap = cv2.VideoCapture("tmp_in.mp4")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for i in np.linspace(0, n - 1, 6).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if ok:
                    frames.append(fr)
        else:
            frames.append(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
        frames = [fr for fr in frames if fr is not None]
        if not frames:
            st.error("Could not read this file as an image or video. Please try a JPG, PNG or MP4.")
        images, dets, failed = [], [], None
        with st.spinner("Detecting vessels (the first request after idle can take ~30 s while the detector wakes up)..."):
            for fr in frames:
                ok, buf = cv2.imencode(".jpg", fr)
                try:
                    r = detect_frame(api, buf.tobytes(), conf_thr)
                except (requests.RequestException, ValueError, KeyError) as e:
                    failed = e
                    break
                dets += r["detections"]
                images.append(base64.b64decode(r["annotated_jpg_b64"]))
        if failed is not None:
            st.error("The detector did not answer. It may still be waking up: wait 30 seconds and upload again. "
                     f"({type(failed).__name__})")
        else:
            st.session_state.update(sig=sig, images=images, dets=dets[:6])

images = st.session_state.get("images", [])
dets = st.session_state.get("dets", [])

left, right = st.columns(2)
with left:
    st.subheader("Detection collage")
    if images:
        st.image(images, width=260)
    else:
        st.info("Upload an image or video to see detections.")
    if dets and st.button("Classify vessel types with Gemini"):
        with st.spinner("Asking Gemini (one request for all vessels)..."):
            res = iter(classify_crops([base64.b64decode(d["crop_b64"]) for d in dets if d.get("crop_b64")]))
        for d in dets:
            d["vessel_type"] = next(res) if d.get("crop_b64") else {}
    if dets:
        st.dataframe(pd.DataFrame([{"yolo_class": d["class"], "conf": d["conf"],
                                    "gemini_type": d.get("vessel_type", {}).get("type", "(not classified)"),
                                    "status": d.get("vessel_type", {}).get("status", ""),
                                    "confidence": d.get("vessel_type", {}).get("confidence", ""),
                                    "reason": d.get("vessel_type", {}).get("reason", "")} for d in dets]),
                     width="stretch")
        errs = [d["vessel_type"]["reason"] for d in dets if d.get("vessel_type", {}).get("reason", "").startswith("Gemini")]
        if errs:
            st.warning(errs[0])
        for i, d in enumerate(dets, 1):
            if d.get("vessel_type", {}).get("colreg_status"):
                colreg_card(f"Camera target {i} (conf {d['conf']:.2f})", d["vessel_type"])
with right:
    st.subheader("Waterway map (SIMULATED AIS)")
    if os.path.exists("assets/waterway.png"):
        st.image("assets/waterway.png")
    own, targets = make_ais()
    st.map(pd.DataFrame([own] + targets)[["lat", "lon"]])

assessment = assess(own, targets)
st.subheader("Collision risk + COLREG (SIMULATED AIS targets)")
st.caption("These AIS targets are simulated for the demo. They are not taken from the uploaded image. "
           "CPA/TCPA and COLREG responsibility are computed by deterministic rules, not by Gemini.")
st.dataframe(pd.DataFrame(assessment)[["mmsi", "class", "status", "cpa_nm", "tcpa_min", "risk",
                                       "colreg_status", "give_way"]], width="stretch")
for row in assessment:
    if row["risk"] != "LOW":
        colreg_card(f"{row['risk']} risk: {row['mmsi']} (CPA {row['cpa_nm']} nm, TCPA {row['tcpa_min']} min)", row)
with st.expander("COLREG guidance for every target"):
    for row in assessment:
        colreg_card(f"{row['mmsi']} ({row['risk']})", row)
if st.button("Ask Gemini for alert + sea-state summary"):
    summary = [{"yolo_class": d["class"], "conf": d["conf"],
                "vessel_type": d.get("vessel_type", {}).get("type", "unclassified"),
                "colreg_status": d.get("vessel_type", {}).get("colreg_status", "")} for d in dets]
    out = explain(assessment, summary, weather)
    st.error(out.get("alert", ""))
    st.write("**Advice:**", out.get("advice", ""))
    st.write("**Sea state:**", out.get("sea_state_summary", ""))
    if out.get("error"):
        st.caption("Gemini unavailable, showing rule-based text: " + out["error"])

st.subheader("Engine telemetry (SIMULATED)")
df = make_telemetry()
st.line_chart(df.set_index("time")[["fuel_lph", "vibration_mm_s"]])
for a in maintenance_alerts(df):
    st.warning(a)

st.subheader("Fuel impact estimate (ASSUMPTION-BASED)")
c1, c2, c3 = st.columns(3)
events = c1.number_input("Avoided emergency maneuvers / month", 0, 100, 6)
per_event = c2.number_input("Fuel penalty per emergency maneuver (L)", 0, 2000, 300)
baseline = c3.number_input("Baseline fuel / month (L)", 1000, 2000000, 250000)
saved, pct = fuel_impact(events, per_event, baseline)
st.metric("Estimated fuel saved / month", f"{saved:,.0f} L", f"{pct:.2f}% of baseline")
st.caption("Illustrative assumptions, not measured results. Replace with real fleet data.")
