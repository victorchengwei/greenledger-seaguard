"""Collision-risk maths (deterministic) + Gemini plain-language explanation.

Design: CPA/TCPA are computed in code. Gemini only explains the numbers and
suggests actions; it never invents data. Decision support only.
"""
import json
import math
import os

from agent.colreg import LEGACY_TYPES, colreg_assess

KN = 0.514444  # knots -> m/s
NM = 1852.0    # metres per nautical mile


def _xy(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111320 * math.cos(math.radians(lat0)),
            (lat - lat0) * 110540)


def _vel(sog, cog):
    v = sog * KN
    return v * math.sin(math.radians(cog)), v * math.cos(math.radians(cog))


def cpa_tcpa(own, tgt):
    """own/tgt: dict(lat, lon, sog [kn], cog [deg]). Returns (cpa_nm, tcpa_min)."""
    tx, ty = _xy(tgt["lat"], tgt["lon"], own["lat"], own["lon"])
    ovx, ovy = _vel(own["sog"], own["cog"])
    tvx, tvy = _vel(tgt["sog"], tgt["cog"])
    rvx, rvy = tvx - ovx, tvy - ovy
    vv = rvx ** 2 + rvy ** 2
    t = 0.0 if vv == 0 else max(0.0, -(tx * rvx + ty * rvy) / vv)
    cpa = math.hypot(tx + rvx * t, ty + rvy * t)
    return cpa / NM, t / 60


def level(cpa_nm, tcpa_min):
    if cpa_nm < 0.5 and tcpa_min < 12:
        return "HIGH"
    if cpa_nm < 1.0 and tcpa_min < 20:
        return "MEDIUM"
    return "LOW"


def assess(own, targets):
    out = []
    for t in targets:
        cpa, tcpa = cpa_tcpa(own, t)
        vtype = LEGACY_TYPES.get(t.get("class"), t.get("class", "Other"))
        status = t.get("status", "underway")
        row = {"mmsi": t.get("mmsi"), "class": vtype, "status": status,
               "cpa_nm": round(cpa, 2), "tcpa_min": round(tcpa, 1),
               "risk": level(cpa, tcpa)}
        row.update(colreg_assess(vtype, status, own, t))
        out.append(row)
    return sorted(out, key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[r["risk"]])


def explain(assessment, detections, weather):
    """Ask Gemini for {alert, advice, sea_state_summary}; fall back to rules."""
    worst = assessment[0] if assessment else None
    fallback = {
        "alert": (f"{worst['risk']} risk with {worst['class']} (CPA {worst['cpa_nm']} nm "
                  f"in {worst['tcpa_min']} min)") if worst else "No traffic detected",
        "advice": (f"{worst['give_way']}. {worst['action']}" if worst and worst["risk"] != "LOW"
                   else "Maintain watch."),
        "sea_state_summary": f"Wind {weather.get('wind_kn')} kn, waves {weather.get('wave_height_m')} m, "
                             f"visibility {weather.get('visibility_nm')} nm.",
    }
    if not os.getenv("GEMINI_API_KEY"):
        return fallback
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()  # reads GEMINI_API_KEY
        prompt = (
            "You are a bridge-watch assistant (decision support only). Use ONLY the data below. "
            "The COLREG fields (colreg_status, give_way, action) come from a rule engine: repeat them, "
            "never change who gives way. "
            "Reply as JSON with keys alert, advice, sea_state_summary: short, plain English.\n"
            f"Risk assessment: {json.dumps(assessment)}\n"
            f"Camera detections: {json.dumps(detections)}\n"
            f"Weather: {json.dumps(weather)}")
        r = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        return json.loads(r.text)
    except Exception as e:  # keep the demo alive if the API fails
        fallback["error"] = str(e)
        return fallback


def fuel_impact(events_avoided_per_month, fuel_per_event_l, baseline_l_per_month):
    """Transparent, assumption-based estimate. Returns (litres_saved, percent_of_baseline).
    Inputs are ASSUMPTIONS to be replaced by real fleet data."""
    saved = events_avoided_per_month * fuel_per_event_l
    return saved, 100.0 * saved / baseline_l_per_month
