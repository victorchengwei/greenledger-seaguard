"""Deterministic COLREG rule engine (decision support only).

Own ship is assumed to be a POWER-DRIVEN VESSEL UNDERWAY. Nothing here replaces a proper
lookout (Rule 5), safe speed (Rule 7) or the officer of the watch's judgement (Rule 2).
"""
import math

STATUSES = ["underway", "engaged_in_fishing", "towing", "restricted_maneuverability",
            "not_under_command", "under_sail", "at_anchor_or_moored",
            "constrained_by_draught", "unknown"]

# Navigational status -> (COLREG status, responsibility of own ship, recommended action)
_STATUS_RULES = {
    "engaged_in_fishing": (
        "從事捕魚中 Engaged in fishing (Rule 26)",
        "本船應避讓 Own ship gives way (Rule 18)",
        "早期、大幅度行動，通常向右轉向從其船尾通過，遠離漁具 "
        "Early & substantial action, usually to starboard, pass astern, clear of gear (Rule 8, 16)"),
    "restricted_maneuverability": (
        "操縱能力受限 Restricted in ability to manoeuvre (Rule 27)",
        "本船應避讓 Own ship gives way (Rule 18)",
        "早期、大幅度行動，保持寬裕距離 Early & substantial action, wide berth (Rule 8, 16)"),
    "not_under_command": (
        "失控船 Not under command (Rule 27)",
        "本船應避讓 Own ship gives way (Rule 18)",
        "早期、大幅度行動，必要時減速或停俥 Early & substantial action, slow down or stop if needed (Rule 8)"),
    "under_sail": (
        "帆船航行中 Sailing vessel (Rule 25)",
        "本船應避讓 Own ship gives way (Rule 18)",
        "早期、大幅度行動，避免從其船首通過 Early & substantial action, avoid crossing ahead (Rule 8, 16)"),
    "constrained_by_draught": (
        "受吃水限制 Constrained by draught (Rule 28)",
        "本船不應妨礙其通過 Do not impede (Rule 18(d))",
        "儘早讓出深水航道 Leave the deep-water route clear early (Rule 8(f))"),
    "towing": (
        "拖帶中 Towing (Rule 24)",
        "拖帶嚴重受限時本船避讓，否則依會遇態勢 Give way if severely restricted (Rule 18), else by encounter",
        "切勿從拖船與被拖物之間通過 Never pass between tug and tow; keep a wide berth"),
    "at_anchor_or_moored": (
        "錨泊或繫泊中 At anchor / moored (Rule 30)",
        "本船保持遠離 Own ship keeps clear",
        "減速並保持安全距離，注意錨鏈 Reduce speed, keep clear, mind the cable"),
}
_UNKNOWN = (
    "狀態不明 Status unknown",
    "謹慎處置 Treat with caution (Rule 2, 7)",
    "確認燈號號標，保持瞭望，備便避讓 Verify lights/shapes, keep lookout, be ready to act (Rule 5, 7)")

# Encounter between two power-driven vessels -> (situation, responsibility, action)
_ENCOUNTER = {
    "head_on": ("對遇 Head-on (Rule 14)",
                "雙方皆應避讓 Both give way (Rule 14)",
                "各自向右轉向，從對方左舷通過 Both alter to starboard, pass port to port"),
    "crossing_give_way": ("交叉相遇，他船在右舷 Crossing, target to starboard (Rule 15)",
                          "本船應避讓 Own ship gives way (Rule 15, 16)",
                          "早期、大幅度向右轉向，避免從其船首通過 Early & substantial alteration to starboard, "
                          "avoid crossing ahead (Rule 8, 15)"),
    "crossing_stand_on": ("交叉相遇，他船在左舷 Crossing, target to port (Rule 15)",
                          "本船直航 Own ship stands on (Rule 17)",
                          "保持航向航速，密切監視；他船未行動時採取行動 "
                          "Keep course & speed, watch closely; act if give-way vessel does not (Rule 17)"),
    "overtaking": ("本船追越他船 Own ship overtaking (Rule 13)",
                   "本船應避讓 Own ship gives way (Rule 13)",
                   "保持遠離被追越船直到完全駛過 Keep well clear until finally past and clear"),
    "being_overtaken": ("他船追越本船 Being overtaken (Rule 13)",
                        "本船直航 Own ship stands on (Rule 17)",
                        "保持航向航速，密切監視 Keep course & speed, watch closely (Rule 17)"),
}

# Vessel-type names used before the 12-type list, mapped to the new names.
LEGACY_TYPES = {"Cargoship": "Cargo/Container", "Fishingboat": "Fishing",
                "Tankership": "Tanker", "Workboat": "Workboat"}


def _bearing(own, tgt):
    """True bearing (deg) from own to target using a flat-earth approximation."""
    dx = (tgt["lon"] - own["lon"]) * 111320 * math.cos(math.radians(own["lat"]))
    dy = (tgt["lat"] - own["lat"]) * 110540
    return math.degrees(math.atan2(dx, dy)) % 360


def encounter(own, tgt):
    """Classify the encounter between two power-driven vessels from AIS position and course."""
    brg = _bearing(own, tgt)
    rel = (brg - own["cog"]) % 360                       # target's bearing relative to own heading
    rel_from_tgt = ((brg + 180) % 360 - tgt["cog"]) % 360  # own ship's bearing relative to target heading
    course_diff = abs(((tgt["cog"] - own["cog"]) + 180) % 360 - 180)
    if course_diff >= 170 and (rel <= 10 or rel >= 350):
        return "head_on"
    if 112.5 < rel_from_tgt < 247.5 and own["sog"] > tgt["sog"]:
        return "overtaking"
    if 112.5 < rel < 247.5 and tgt["sog"] > own["sog"]:
        return "being_overtaken"
    return "crossing_give_way" if rel < 180 else "crossing_stand_on"


def colreg_assess(vtype="Other", status="unknown", own=None, tgt=None):
    """Return {"colreg_status", "give_way", "action"} for one target vessel.

    With own/tgt AIS data, power-driven encounters are resolved by Rules 13-15.
    Without it (camera only), the answer says that a bearing is needed."""
    status = status if status in STATUSES else "unknown"
    if status in _STATUS_RULES:
        s, g, a = _STATUS_RULES[status]
    elif status == "underway":
        if own and tgt:
            s, g, a = _ENCOUNTER[encounter(own, tgt)]
        else:
            s = "機動船航行中 Power-driven vessel underway (Rule 23)"
            g = "依會遇態勢判斷 By encounter (Rule 13-15), needs AIS/radar bearing"
            a = "取得方位後判斷；對遇或他船在右舷時通常向右轉向 Get a bearing; usually starboard if head-on or target to starboard"
        if vtype == "Fishing":
            s += "；漁船未作業視為機動船 Fishing vessel not engaged = power-driven"
    else:
        s, g, a = _UNKNOWN
    return {"colreg_status": s, "give_way": g, "action": a}
