"""Stage 2 (optional): Gemini names the vessel type AND visible navigational status for ALL
crops in ONE request; the COLREG rule engine (agent/colreg.py) then adds guidance.

One request instead of one per vessel keeps us inside the free-tier quota.
Retries with a pause on 429/503.
"""
import json
import os
import time

from agent.colreg import STATUSES, colreg_assess

TYPES = ["Cargo/Container", "Tanker", "Bulk Carrier", "Fishing", "Workboat", "Tug",
         "Pilot", "Rescue/SAR", "Passenger/Ferry", "Naval", "Yacht/Sailing", "Other"]


def _finish(o):
    """Validate one Gemini answer and attach the COLREG fields."""
    if o.get("type") not in TYPES + ["Unknown"]:
        o["type"] = "Other"
    if o.get("status") not in STATUSES:
        o["status"] = "unknown"
    o.update(colreg_assess(o["type"], o["status"]))
    return o


def classify_crops(jpg_list, retries=3):
    """jpg_list: list of JPEG bytes. Returns a list of
    {"type","confidence","status","reason","colreg_status","give_way","action"}."""
    n = len(jpg_list)
    if n == 0:
        return []

    def unknown(why):
        return [_finish({"type": "Unknown", "confidence": "-", "status": "unknown", "reason": why})
                for _ in range(n)]

    if not os.getenv("GEMINI_API_KEY"):
        return unknown("No GEMINI_API_KEY set")
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
    except Exception as e:
        return unknown(f"Gemini setup error: {e}")

    parts = []
    for i, b in enumerate(jpg_list, 1):
        parts += [f"Image {i}:", types.Part.from_bytes(data=b, mime_type="image/jpeg")]
    prompt = (
        f"You are a ship's officer. For each of the {n} images above, classify the vessel as exactly "
        f"one of {TYPES}, and choose its navigational status as exactly one of {STATUSES}. "
        "Choose a status only from visible evidence: nets or fishing gear deployed, a towline, crane or "
        "dredging work, sails set, an anchor ball, day shapes. With no clear evidence use 'underway' for "
        "a vessel making way, otherwise 'unknown'. "
        f"Reply as a JSON list of {n} objects, in image order: "
        '{"type": "...", "confidence": "low|medium|high", "status": "...", '
        '"reason": "one short sentence citing the visual evidence"}')
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                contents=parts + [prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            out = json.loads(r.text)
            if isinstance(out, dict):
                out = [out]
            out = [o for o in out if isinstance(o, dict)]
            return ([_finish(o) for o in out] + unknown("missing from reply"))[:n]
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "503" in msg) and attempt < retries - 1:
                time.sleep(25 * (attempt + 1))
                continue
            return unknown(f"Gemini error: {msg[:200]}")
    return unknown("Gemini error: retries exhausted")
