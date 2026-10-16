"""Gemini output is untrusted: it is validated before the rule engine sees it.
These tests replace the Gemini client with a fake, so they run offline and free."""
import json
import sys
import types

import pytest

import agent.vessel_type as vt


def install_fake_gemini(monkeypatch, reply_text=None, error=None):
    def generate_content(**_):
        if error:
            raise error
        return types.SimpleNamespace(text=reply_text)

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    gtypes = types.SimpleNamespace(Part=types.SimpleNamespace(from_bytes=lambda **_: "img"),
                                   GenerateContentConfig=lambda **_: None)
    genai = types.SimpleNamespace(Client=lambda: client, types=gtypes)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(vt.time, "sleep", lambda _: None)


def test_valid_reply_gets_colreg_fields(monkeypatch):
    install_fake_gemini(monkeypatch, json.dumps([
        {"type": "Fishing", "confidence": "high", "status": "engaged_in_fishing", "reason": "nets out"}]))
    [out] = vt.classify_crops([b"jpg"])
    assert out["type"] == "Fishing"
    assert "Rule 26" in out["colreg_status"]


def test_invented_labels_are_clamped(monkeypatch):
    install_fake_gemini(monkeypatch, json.dumps([{"type": "Spaceship", "status": "flying"}]))
    [out] = vt.classify_crops([b"jpg"])
    assert out["type"] == "Other"
    assert out["status"] == "unknown"


def test_short_reply_is_padded_to_one_result_per_crop(monkeypatch):
    install_fake_gemini(monkeypatch, json.dumps({"type": "Tug", "status": "towing"}))
    out = vt.classify_crops([b"a", b"b", b"c"])
    assert [o["type"] for o in out] == ["Tug", "Unknown", "Unknown"]


def test_quota_error_degrades_gracefully(monkeypatch):
    install_fake_gemini(monkeypatch, error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    out = vt.classify_crops([b"a", b"b"])
    assert all(o["type"] == "Unknown" and o["reason"].startswith("Gemini error") for o in out)


def test_no_key_means_no_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    [out] = vt.classify_crops([b"a"])
    assert out["reason"] == "No GEMINI_API_KEY set"


@pytest.mark.parametrize("crops", [[]])
def test_no_crops_no_work(crops):
    assert vt.classify_crops(crops) == []
