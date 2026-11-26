"""Indic Parler-TTS and MMS-TTS Malayalam: registry wiring, opt-in gating,
adapter contract, licence gating and the shared sidecar's pure logic. No GPU,
no upstream package: the sidecar is driven with stub parts."""
import base64
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture
def sc():
    spec = importlib.util.spec_from_file_location(
        "indic_parler_sidecar_main", ROOT / "backend/engines/indic_parler/main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frames(raw: bytes) -> list[dict]:
    out, buf = [], io.BytesIO(raw)
    while (header := buf.read(4)) and len(header) == 4:
        (n,) = struct.unpack("!I", header)
        out.append(json.loads(buf.read(n)))
    return out


def _registry() -> dict:
    return json.loads((ROOT / "backend/config/model_licenses.json").read_text("utf-8"))


# ── wiring ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("engine_id, cls", [
    ("indic-parler-tts", "IndicParlerBackend"),
    ("mms-tts-mal", "MMSMalayalamBackend"),
])
def test_registered_everywhere_the_app_looks(engine_id, cls):
    from services.tts_backend import (
        _ENGINE_DOCS, _INSTALL_HINTS, _LAZY_REGISTRY, _SETUP_SNIPPETS,
    )

    assert _LAZY_REGISTRY[engine_id] == ("engines.indic_parler", cls)
    assert "OMNIVOICE_INDIC_PARLER_DIR" in _SETUP_SNIPPETS[engine_id]
    assert (ROOT / _ENGINE_DOCS[engine_id]).is_file()
    assert "malayalam" in _INSTALL_HINTS[engine_id].lower()


def test_one_install_backs_both_engines():
    from services.sidecar_install import SPECS

    spec = SPECS["indic-parler-tts"]
    assert spec.env_var == "OMNIVOICE_INDIC_PARLER_DIR"
    assert spec.source_revision and spec.source_revision in spec.tarball_url
    assert "mms-tts-mal" not in SPECS  # shares the Indic Parler-TTS venv


def test_engine_metadata():
    from engines.indic_parler import IndicParlerBackend, MMSMalayalamBackend

    assert IndicParlerBackend.supports_cloning is False
    assert IndicParlerBackend.supports_voice_design is True
    assert IndicParlerBackend.requires_hf_auth is True
    parler_langs = IndicParlerBackend.supported_languages.fget(None)
    assert "ml" in parler_langs and len(parler_langs) == 21
    assert MMSMalayalamBackend.supports_cloning is False
    assert MMSMalayalamBackend.supports_voice_design is False
    assert MMSMalayalamBackend.supported_languages.fget(None) == ["ml"]


def test_inert_without_install(monkeypatch, tmp_path):
    monkeypatch.delenv("OMNIVOICE_INDIC_PARLER_DIR", raising=False)
    from engines.indic_parler import IndicParlerBackend, MMSMalayalamBackend, bootstrap

    monkeypatch.setattr(bootstrap, "_ENGINES_VENV_DIR", tmp_path / "none")
    bootstrap.invalidate()
    for cls in (IndicParlerBackend, MMSMalayalamBackend):
        ok, reason = cls.is_available()
        assert ok is False and "Model Catalogue" in reason
    with pytest.raises(RuntimeError, match="OMNIVOICE_INDIC_PARLER_DIR"):
        bootstrap.resolve_indic_parler_venv()


def test_each_engine_tells_the_shared_sidecar_which_model():
    from engines.indic_parler import IndicParlerBackend, MMSMalayalamBackend

    parler = IndicParlerBackend.__new__(IndicParlerBackend)
    mms = MMSMalayalamBackend.__new__(MMSMalayalamBackend)
    assert parler.extra_spawn_env()["VOICESTUDIO_INDIC_PARLER_MODE"] == "parler"
    assert mms.extra_spawn_env()["VOICESTUDIO_INDIC_PARLER_MODE"] == "mms"


def test_pins_match_the_licence_registry(sc):
    rows = {m["id"]: m for m in _registry()["models"]}
    assert rows[sc.PARLER_MODEL]["revision"] == sc.PARLER_REVISION
    assert rows[sc.DESCRIPTION_TOKENIZER]["revision"] == sc.DESCRIPTION_TOKENIZER_REVISION
    assert rows[sc.MMS_MODEL]["revision"] == sc.MMS_REVISION
    assert rows[sc.MMS_MODEL]["engines"] == ["mms-tts-mal"]


def test_non_commercial_mms_requires_licence_acceptance():
    from services.model_licenses import license_category

    row = next(m for m in _registry()["models"] if m["id"] == "facebook/mms-tts-mal")
    assert license_category(row) != "commercial"


# ── adapter contract ──────────────────────────────────────────────────────


def _capture(monkeypatch):
    from services.subprocess_backend import SubprocessBackend

    sent = {}

    def fake_generate(self, text, **kw):
        sent.update(kw, text=text)
        return "audio"

    monkeypatch.setattr(SubprocessBackend, "generate", fake_generate)
    return sent


def test_parler_forwards_the_voice_description(monkeypatch):
    from engines.indic_parler import IndicParlerBackend

    sent = _capture(monkeypatch)
    backend = IndicParlerBackend.__new__(IndicParlerBackend)
    backend.generate("നമസ്കാരം", instruct=" Harish speaks slowly. ", language="ml",
                     ref_audio="ignored.wav")
    assert sent == {"text": "നമസ്കാരം", "description": "Harish speaks slowly.",
                    "language": "ml"}


def test_parler_defaults_to_a_named_malayalam_speaker(monkeypatch):
    from engines.indic_parler import IndicParlerBackend

    sent = _capture(monkeypatch)
    IndicParlerBackend.__new__(IndicParlerBackend).generate("നമസ്കാരം", language="ml")
    assert sent["description"].startswith("Anjali ")


def test_mms_speaks_malayalam_only(monkeypatch):
    from engines.indic_parler import MMSMalayalamBackend

    sent = _capture(monkeypatch)
    backend = MMSMalayalamBackend.__new__(MMSMalayalamBackend)
    backend.generate("നമസ്കാരം", instruct="ignored", language="ml")
    assert sent == {"text": "നമസ്കാരം", "language": "ml"}
    sent.clear()
    with pytest.raises(ValueError):
        backend.generate("hello", language="en")
    assert not sent


# ── sidecar ───────────────────────────────────────────────────────────────


def test_mode_defaults_to_parler(sc, monkeypatch):
    monkeypatch.delenv("VOICESTUDIO_INDIC_PARLER_MODE", raising=False)
    assert sc._mode() == "parler"
    monkeypatch.setenv("VOICESTUDIO_INDIC_PARLER_MODE", "MMS")
    assert sc._mode() == "mms"
    monkeypatch.setenv("VOICESTUDIO_INDIC_PARLER_MODE", "other")
    assert sc._mode() == "parler"


def test_parler_needs_a_description(sc, monkeypatch):
    monkeypatch.setenv("VOICESTUDIO_INDIC_PARLER_MODE", "parler")
    monkeypatch.setattr(sc, "_load", lambda out: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="description"):
        sc._handle_synthesize({"text": "x", "description": "  "}, io.BytesIO())


def test_synthesize_with_stub_parts(sc, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("VOICESTUDIO_INDIC_PARLER_MODE", "mms")
    calls = []

    def fake_synth(parts, text, description):
        calls.append((text, description))
        return np.array([[0.0, 0.5, -1.0]], dtype=np.float32)

    monkeypatch.setattr(sc, "_load", lambda out: {"mode": "mms", "model": None})
    monkeypatch.setattr(sc, "_synthesize", fake_synth)
    out = io.BytesIO()
    sc._handle_synthesize({"text": "നമസ്കാരം"}, out)
    (frame,) = _frames(out.getvalue())
    assert calls == [("നമസ്കാരം", "")]
    assert frame["op"] == "audio" and frame["sample_rate"] == 16000
    pcm = np.frombuffer(base64.b64decode(frame["audio_pcm_b64"]), dtype="<i2")
    assert pcm.tolist() == [0, 16383, -32767]


def test_gated_errors_explain_the_fix(sc):
    err = sc._gated_hint(OSError("401 Client Error: gated repo"), sc.PARLER_MODEL)
    assert isinstance(err, RuntimeError) and "accept" in str(err)
    plain = ValueError("boom")
    assert sc._gated_hint(plain, sc.PARLER_MODEL) is plain


def test_sidecar_imports_no_parent_code():
    source = (ROOT / "backend/engines/indic_parler/main.py").read_text("utf-8")
    assert "from services" not in source and "from core" not in source
