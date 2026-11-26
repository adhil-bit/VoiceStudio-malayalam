"""The CI reel renderer speaks every line with the repo's own sidecar code and
mixes them; offline, with the model stubbed."""
import importlib.util
import shutil
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("render_reel_ci", ROOT / "scripts/render_reel_ci.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_speaks_each_line_and_writes_the_voice_track(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    mod = _mod()
    real_load = mod._load
    spoken = []

    def load(name, path):
        m = real_load(name, path)
        if name == "indic_parler_sidecar":
            m._load = lambda out: {"mode": "mms", "model": None}
            m._sample_rate = lambda parts: 16000
            m._synthesize = lambda parts, text, desc: spoken.append(text) or np.full(8000, 0.2, np.float32)
        return m

    monkeypatch.setattr(mod, "_load", load)
    assert mod.main(["--engine", "mms", "--out", str(tmp_path)]) == 0
    assert len(spoken) == 6 and spoken[0].startswith("ഒരു")
    with wave.open(str(tmp_path / "narration.wav")) as w:
        assert w.getnframes() / w.getframerate() > 36  # last line starts at 36.2 s
