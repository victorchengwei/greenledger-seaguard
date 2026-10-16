"""The COLREG engine is deterministic, so every rule it applies is pinned by a test."""
import pytest

from agent.colreg import STATUSES, colreg_assess, encounter

OWN = {"lat": 0.0, "lon": 0.0, "sog": 10, "cog": 0}   # own ship heading north at 10 kn


@pytest.mark.parametrize("target, expected", [
    ({"lat": 0.05, "lon": 0.0, "sog": 10, "cog": 180}, "head_on"),             # dead ahead, reciprocal course
    ({"lat": 0.02, "lon": 0.03, "sog": 10, "cog": 270}, "crossing_give_way"),  # on our starboard bow
    ({"lat": 0.02, "lon": -0.03, "sog": 10, "cog": 90}, "crossing_stand_on"),  # on our port bow
    ({"lat": 0.02, "lon": 0.0, "sog": 5, "cog": 0}, "overtaking"),             # slower ship ahead, same course
    ({"lat": -0.02, "lon": 0.0, "sog": 15, "cog": 0}, "being_overtaken"),      # faster ship astern
])
def test_encounter_geometry(target, expected):
    assert encounter(OWN, target) == expected


@pytest.mark.parametrize("status, rule, gives_way", [
    ("engaged_in_fishing", "Rule 26", "Own ship gives way (Rule 18)"),
    ("restricted_maneuverability", "Rule 27", "Own ship gives way (Rule 18)"),
    ("not_under_command", "Rule 27", "Own ship gives way (Rule 18)"),
    ("under_sail", "Rule 25", "Own ship gives way (Rule 18)"),
    ("constrained_by_draught", "Rule 28", "Do not impede (Rule 18(d))"),
])
def test_rule_18_hierarchy(status, rule, gives_way):
    out = colreg_assess("Other", status)
    assert rule in out["colreg_status"]
    assert gives_way in out["give_way"]


def test_head_on_both_alter_to_starboard():
    out = colreg_assess("Cargo/Container", "underway", OWN, {"lat": 0.05, "lon": 0.0, "sog": 10, "cog": 180})
    assert "Rule 14" in out["colreg_status"]
    assert "Both give way" in out["give_way"]
    assert "starboard" in out["action"]


def test_crossing_from_port_means_stand_on():
    out = colreg_assess("Tanker", "underway", OWN, {"lat": 0.02, "lon": -0.03, "sog": 10, "cog": 90})
    assert "stands on (Rule 17)" in out["give_way"]


def test_camera_only_target_asks_for_a_bearing():
    """Without AIS/radar we cannot know the encounter, so the engine must not guess."""
    out = colreg_assess("Cargo/Container", "underway")
    assert "needs AIS/radar bearing" in out["give_way"]


def test_fishing_vessel_not_fishing_is_power_driven():
    out = colreg_assess("Fishing", "underway")
    assert "Fishing vessel not engaged = power-driven" in out["colreg_status"]


def test_towing_never_pass_between_tug_and_tow():
    assert "Never pass between tug and tow" in colreg_assess("Tug", "towing")["action"]


@pytest.mark.parametrize("status", ["unknown", "made_up_status", "", None])
def test_unknown_status_is_handled_cautiously(status):
    out = colreg_assess("Other", status)
    assert "Treat with caution" in out["give_way"]


def test_every_status_returns_all_three_fields():
    for status in STATUSES:
        out = colreg_assess("Other", status)
        assert set(out) == {"colreg_status", "give_way", "action"}
        assert all(out.values())
