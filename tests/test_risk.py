"""CPA/TCPA maths and the risk table built from simulated AIS."""
import math

import pytest

from agent.risk import assess, cpa_tcpa, explain, fuel_impact, level
from sim.simulate import make_ais

OWN = {"lat": 22.6, "lon": 120.25, "sog": 12, "cog": 90}


def test_collision_course_gives_zero_cpa():
    # Target 2 nm due east, heading west: closing speed 12 + 12 = 24 kn -> meet in 5 minutes.
    dlon = 2 * 1852 / (111320 * math.cos(math.radians(22.6)))
    tgt = {"lat": 22.6, "lon": 120.25 + dlon, "sog": 12, "cog": 270}
    cpa, tcpa = cpa_tcpa(OWN, tgt)
    assert cpa == pytest.approx(0.0, abs=0.01)
    assert tcpa == pytest.approx(5.0, abs=0.1)


def test_diverging_target_has_tcpa_zero():
    tgt = {"lat": 22.6, "lon": 120.20, "sog": 12, "cog": 270}   # behind us, moving away
    _, tcpa = cpa_tcpa(OWN, tgt)
    assert tcpa == 0.0


@pytest.mark.parametrize("cpa, tcpa, expected", [
    (0.3, 6, "HIGH"), (0.8, 15, "MEDIUM"), (0.3, 30, "LOW"), (2.0, 5, "LOW"),
])
def test_risk_levels(cpa, tcpa, expected):
    assert level(cpa, tcpa) == expected


def test_assessment_is_sorted_and_carries_colreg_fields():
    rows = assess(*make_ais())
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    assert [order[r["risk"]] for r in rows] == sorted(order[r["risk"]] for r in rows)
    for r in rows:
        assert {"colreg_status", "give_way", "action", "status"} <= set(r)


def test_demo_scenario_matches_the_pitch():
    rows = {r["mmsi"]: r for r in assess(*make_ais())}
    assert rows["SIM001"]["risk"] == "HIGH" and "Rule 14" in rows["SIM001"]["colreg_status"]
    assert rows["SIM002"]["risk"] == "MEDIUM" and "Rule 26" in rows["SIM002"]["colreg_status"]


def test_explain_falls_back_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rows = assess(*make_ais())
    out = explain(rows, [], {"wind_kn": 18, "wave_height_m": 1.5, "visibility_nm": 1.0})
    assert "HIGH" in out["alert"]
    assert "Rule 14" in out["advice"]


def test_fuel_impact_is_transparent_arithmetic():
    saved, pct = fuel_impact(6, 300, 250000)
    assert saved == 1800
    assert pct == pytest.approx(0.72)
