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


def test_builtin_script_matches_the_reel_scenes():
    assert [round(l["at"]) for l in mod.SPOTLIGHT_REEL] == [0, 5, 13, 20, 30, 36]
