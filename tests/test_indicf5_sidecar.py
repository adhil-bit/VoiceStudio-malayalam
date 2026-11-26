"""IndicF5 engine (Malayalam + 10 Indian languages): registry wiring, opt-in
gating, adapter contract and the sidecar's pure logic. No GPU, no upstream
package: the sidecar is driven with a stub model."""
import base64
import importlib.util
import io
import json
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture
def sc():
    spec = importlib.util.spec_from_file_location(
        "indicf5_sidecar_main", ROOT / "backend" / "engines" / "indicf5" / "main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frames(raw: bytes) -> list[dict]:
    out, buf = [], io.BytesIO(raw)
    while True:
        header = buf.read(4)
        if len(header) < 4:
            return out
        (n,) = struct.unpack("!I", header)
        out.append(json.loads(buf.read(n)))


# ── wiring ────────────────────────────────────────────────────────────────


def test_registered_everywhere_the_app_looks():
    from services.tts_backend import (
        _ENGINE_DOCS, _INSTALL_HINTS, _LAZY_REGISTRY, _SETUP_SNIPPETS,
    )
    from services.sidecar_install import SPECS

    assert _LAZY_REGISTRY["indicf5"] == ("engines.indicf5", "IndicF5Backend")
    assert "OMNIVOICE_INDICF5_DIR" in _SETUP_SNIPPETS["indicf5"]
    assert (ROOT / _ENGINE_DOCS["indicf5"]).is_file()
    hint = _INSTALL_HINTS["indicf5"].lower()
    assert "malayalam" in hint
    spec = SPECS["indicf5"]
    assert spec.env_var == "OMNIVOICE_INDICF5_DIR"
    assert spec.source_revision and len(spec.source_revision) == 40
    assert spec.source_revision in spec.tarball_url


def test_backend_metadata_declares_malayalam():
    from engines.indicf5 import IndicF5Backend

    assert IndicF5Backend.id == "indicf5"
    assert IndicF5Backend.requires_hf_auth is True
    assert IndicF5Backend.supports_voice_design is False
    langs = IndicF5Backend.supported_languages.fget(None)
    assert "ml" in langs and len(langs) == 11
    assert "en" not in langs  # upstream lists Indian languages only


def test_inert_without_clone_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("OMNIVOICE_INDICF5_DIR", raising=False)
    from engines.indicf5 import IndicF5Backend, bootstrap

    monkeypatch.setattr(bootstrap, "_ENGINES_VENV_DIR", tmp_path / "none")
    bootstrap.invalidate()
    assert bootstrap.is_indicf5_installed() is False
    ok, reason = IndicF5Backend.is_available()
    assert ok is False
    assert "OMNIVOICE_INDICF5_DIR" in reason and "Hugging Face" in reason
    with pytest.raises(RuntimeError, match="OMNIVOICE_INDICF5_DIR"):
        bootstrap.resolve_indicf5_venv()


def test_pinned_revision_matches_licence_registry(sc):
    registry = json.loads((ROOT / "backend/config/model_licenses.json").read_text("utf-8"))
    record = next(m for m in registry["models"] if m["id"] == sc._DEFAULT_MODEL)
    assert record["revision"] == sc._DEFAULT_REVISION
    assert record["engines"] == ["indicf5"]


# ── adapter contract ──────────────────────────────────────────────────────


def _backend(monkeypatch):
    from engines.indicf5 import IndicF5Backend
    from services.subprocess_backend import SubprocessBackend

    sent = {}

    def fake_generate(self, text, **kw):
        sent.update(kw, text=text)
        return "audio"

    monkeypatch.setattr(SubprocessBackend, "generate", fake_generate)
    return IndicF5Backend.__new__(IndicF5Backend), sent


def test_generate_forwards_reference_and_transcript(monkeypatch, tmp_path):
    backend, sent = _backend(monkeypatch)
    clip = tmp_path / "ref.wav"
    assert backend.generate(
        "നമസ്കാരം", ref_audio=str(clip), ref_text="  എന്റെ ശബ്ദം ", language="ml",
    ) == "audio"
    assert sent == {
        "text": "നമസ്കാരം", "ref_audio": str(clip.resolve()),
        "ref_text": "എന്റെ ശബ്ദം", "language": "ml",
    }


@pytest.mark.parametrize("kw, match", [
    ({"ref_text": "x"}, "reference clip"),
    ({"ref_audio": "a.wav"}, "transcript"),
    ({"ref_audio": "a.wav", "ref_text": "   "}, "transcript"),
])
def test_generate_rejects_missing_inputs_with_clear_message(monkeypatch, kw, match):
    backend, sent = _backend(monkeypatch)
    with pytest.raises(RuntimeError, match=match):
        backend.generate("നമസ്കാരം", **kw)
    assert not sent


def test_generate_rejects_unsupported_language(monkeypatch):
    backend, sent = _backend(monkeypatch)
    with pytest.raises(ValueError):
        backend.generate("hello", ref_audio="a.wav", ref_text="x", language="en")
    assert not sent


# ── sidecar ───────────────────────────────────────────────────────────────


def test_model_source_pins_default_and_requires_sha_for_custom(sc, monkeypatch, tmp_path):
    monkeypatch.delenv("OMNIVOICE_INDICF5_MODEL", raising=False)
    assert sc._model_source() == ("ai4bharat/IndicF5", sc._DEFAULT_REVISION)
    monkeypatch.setenv("OMNIVOICE_INDICF5_MODEL", "someone/fork")
    monkeypatch.setenv("OMNIVOICE_INDICF5_REVISION", "main")
    with pytest.raises(RuntimeError, match="40-character"):
        sc._model_source()
    monkeypatch.setenv("OMNIVOICE_INDICF5_MODEL", str(tmp_path))
    assert sc._model_source() == (str(tmp_path), None)


def test_synthesize_with_stub_model(sc, monkeypatch):
    np = pytest.importorskip("numpy")
    calls = []

    def model(text, ref_audio_path, ref_text):
        calls.append((text, ref_audio_path, ref_text))
        return np.array([0, 16384, -32768], dtype=np.int16)

    monkeypatch.setattr(sc, "_model", model)
    out = io.BytesIO()
    sc._handle_synthesize(
        {"text": "നമസ്കാരം", "ref_audio": "/r.wav", "ref_text": " ശബ്ദം "}, out,
    )
    (frame,) = _frames(out.getvalue())
    assert calls == [("നമസ്കാരം", "/r.wav", "ശബ്ദം")]
    assert frame["op"] == "audio" and frame["sample_rate"] == 24000
    pcm = np.frombuffer(base64.b64decode(frame["audio_pcm_b64"]), dtype="<i2")
    assert pcm.tolist() == [0, 16383, -32767]


def test_synthesize_never_lets_upstream_transcribe(sc, monkeypatch):
    monkeypatch.setattr(sc, "_model", lambda *a, **k: pytest.fail("model must not run"))
    with pytest.raises(ValueError, match="ref_text"):
        sc._handle_synthesize({"text": "x", "ref_audio": "/r.wav"}, io.BytesIO())


def test_sidecar_imports_no_parent_code():
    source = (ROOT / "backend/engines/indicf5/main.py").read_text("utf-8")
    assert "from services" not in source and "from core" not in source
    assert "os.environ)" not in source
