"""The Malayalam reference-voice downloader picks one clean clip per speaker
and records attribution. Offline: the viewer API is faked."""
import importlib.util
import io
import json
import struct
import wave
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location(
        "malayalam_reference_voices", ROOT / "scripts/malayalam_reference_voices.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _wav(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buf.getvalue()


ML = "മലയാളം ഒരു മനോഹരമായ ഭാഷയാണ് എന്ന് എല്ലാവർക്കും അറിയാം"


def _fake(rows, clips):
    def fetch(url):
        if "/rows?" in url:
            q = parse_qs(urlparse(url).query)
            off = int(q["offset"][0])
            return json.dumps({"rows": [{"row": r} for r in rows[off:off + 100]]}).encode()
        return clips[url]
    return fetch


def test_one_clean_clip_per_speaker(mod):
    rows = [
        {"speaker": "A", "text": ML, "audio": [{"src": "u/short", "type": "audio/wav"}]},
        {"speaker": "A", "text": "English only sentence here ok", "audio": [{"src": "u/a1", "type": "audio/wav"}]},
        {"speaker": "A", "text": ML, "audio": [{"src": "u/a2", "type": "audio/wav"}]},
        {"speaker": "A", "text": ML, "audio": [{"src": "u/a3", "type": "audio/wav"}]},
        {"speaker": "B", "text": ML, "audio": [{"src": "u/b1", "type": "audio/mpeg"}]},
        {"speaker": "B", "text": ML, "audio": [{"src": "u/b2", "type": "audio/wav"}]},
    ]
    clips = {"u/short": _wav(2), "u/a1": _wav(8), "u/a2": _wav(9),
             "u/a3": _wav(9), "u/b2": _wav(12)}
    got = list(mod.collect("d", "default", "train", "text", "speaker",
                           per_speaker=1, max_speakers=8, max_rows=500,
                           fetch=_fake(rows, clips)))
    assert [(s, n, round(sec)) for s, n, _t, _d, sec in got] == [("A", 1, 9), ("B", 1, 12)]
    assert got[0][3] == clips["u/a2"]


def test_main_writes_profiles_and_attribution(mod, monkeypatch, tmp_path):
    rows = [{"speaker": "Joji K", "text": ML, "audio": [{"src": "u/1", "type": "audio/wav"}]}]
    monkeypatch.setattr(mod, "_get", _fake(rows, {"u/1": _wav(7)}))
    out = tmp_path / "voices"
    assert mod.main(["--out", str(out)]) == 0
    assert (out / "Joji-K-1.wav").is_file()
    assert (out / "Joji-K-1.txt").read_text("utf-8").strip() == ML
    meta = json.loads((out / "voices.json").read_text("utf-8"))
    assert meta["licence"] == "CC BY-SA 4.0" and meta["clips"][0]["transcript"] == ML
    credit = (out / "ATTRIBUTION.md").read_text("utf-8")
    assert "IMaSC" in credit and "CC BY-SA 4.0" in credit


def test_no_clips_is_a_clear_failure(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_get", _fake([], {}))
    assert mod.main(["--out", str(tmp_path / "v")]) == 1
