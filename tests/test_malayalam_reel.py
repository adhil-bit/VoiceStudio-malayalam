"""The reel narrator builds one ffmpeg mix with every line at its start time."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("malayalam_reel", ROOT / "scripts/malayalam_reel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_lines_are_delayed_to_their_start_and_mixed_over_the_music():
    cmd = mod.mix_command("ffmpeg", [(0.2, Path("a.wav")), (5.2, Path("b.wav"))],
                          Path("out.mp4"), Path("reel.mp4"), 0.3)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[1:a]aresample=48000,adelay=200|200[v0]" in graph
    assert "[2:a]aresample=48000,adelay=5200|5200[v1]" in graph
    assert "[0:a]volume=0.3[music]" in graph and cmd[-1] == "out.mp4"
    assert ["-c:v", "copy"] == cmd[cmd.index("-c:v"):cmd.index("-c:v") + 2]


def test_multipart_carries_malayalam_text():
    body, ctype = mod.multipart({"text": "നമസ്കാരം", "language": "ml", "skip": None}, {})
    assert ctype.startswith("multipart/form-data; boundary=")
    assert "നമസ്കാരം".encode() in body and b'name="skip"' not in body


def test_builtin_script_speaks_the_words_on_screen_in_order():
    ats = [l["at"] for l in mod.SPOTLIGHT_REEL]
    assert ats == sorted(ats) and ats[0] < 0.5 and ats[-1] < mod.REEL_SECONDS
    scenes = (ROOT / "scripts/malayalam_reel.py").read_text("utf-8")
    assert "ഉറപ്പിച്ചു" not in scenes  # no narration that the reel doesn't show


def _wav(path, seconds):
    import struct, wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * int(seconds * 16000))
    return path


def test_long_lines_are_sped_up_to_fit_before_the_next(tmp_path):
    clips = [(0.0, _wav(tmp_path / "a.wav", 3.0)), (2.0, _wav(tmp_path / "b.wav", 1.0)),
             (4.0, _wav(tmp_path / "c.wav", 9.0))]
    tempos = mod.fit_tempos(clips, end=6.0)
    assert tempos[0] > 1.5 - 0.11 and tempos[0] <= mod.MAX_SPEEDUP  # capped
    assert tempos[1] == 1.0 and tempos[2] == mod.MAX_SPEEDUP
    cmd = mod.mix_command("ffmpeg", clips, tmp_path / "o.wav", None, 0.3, tempos)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert f"atempo={mod.MAX_SPEEDUP}" in graph and graph.count("atempo") == 2
