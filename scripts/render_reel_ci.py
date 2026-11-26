#!/usr/bin/env python3
"""Render the Malayalam reel narration without the desktop app (GitHub Actions).

Runs VoiceStudio's own Indic Parler-TTS / MMS-TTS sidecar code
(``backend/engines/indic_parler/main.py``) in-process, at the same pinned model
revisions, speaks every line of the reel script, and mixes the voice into the
video with ``scripts/malayalam_reel.py``'s mixer.

    python scripts/render_reel_ci.py --engine parler --video reels/reel.mp4 --out out/
    python scripts/render_reel_ci.py --engine mms --out out/      # voice track only

``parler`` needs a Hugging Face token (``HF_TOKEN``) whose account accepted the
terms at huggingface.co/ai4bharat/indic-parler-tts. ``mms`` (Meta MMS-TTS
Malayalam, CC-BY-NC-4.0: non-commercial) needs none.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_wav(path: Path, audio, rate: int) -> float:
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().to("cpu").float().numpy()
    arr = np.clip(np.asarray(audio, dtype=np.float32).squeeze(), -1.0, 1.0)
    while arr.ndim > 1:
        arr = arr.mean(axis=int(np.argmin(arr.shape)))
    peak = float(np.abs(arr).max()) or 1.0
    arr = arr * min(1.0, 0.95 / peak) if peak > 0.95 else arr
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((arr * 32767).astype("<i2").tobytes())
    return len(arr) / rate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--engine", choices=("parler", "mms"), default="parler")
    p.add_argument("--video", type=Path, help="the exported reel; omitted or missing: voice track only")
    p.add_argument("--script", type=Path, help='JSON list of {"at": seconds, "text": "..."}')
    p.add_argument("--description", help="Indic Parler-TTS voice description")
    p.add_argument("--music-volume", type=float, default=0.3)
    p.add_argument("--out", type=Path, default=Path("out"))
    args = p.parse_args(argv)

    reel = _load("malayalam_reel", ROOT / "scripts" / "malayalam_reel.py")
    os.environ["VOICESTUDIO_INDIC_PARLER_MODE"] = args.engine
    sc = _load("indic_parler_sidecar", ROOT / "backend" / "engines" / "indic_parler" / "main.py")

    lines = json.loads(args.script.read_text("utf-8")) if args.script else reel.SPOTLIGHT_REEL
    description = args.description or reel.DEFAULT_DESCRIPTION
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading {sc.MMS_MODEL if args.engine == 'mms' else sc.PARLER_MODEL} …", flush=True)
    parts = sc._load(io.BytesIO())
    rate = sc._sample_rate(parts)

    clips = []
    for n, line in enumerate(lines, 1):
        audio = sc._synthesize(parts, line["text"], description)
        path = args.out / f"line-{n:02d}.wav"
        secs = write_wav(path, audio, rate)
        print(f"[{n}/{len(lines)}] at {line['at']:.1f}s, {secs:.1f}s long: {line['text']}", flush=True)
        clips.append((float(line["at"]), path))

    ffmpeg = shutil.which("ffmpeg") or sys.exit("ffmpeg not found")
    end = reel.REEL_SECONDS if not args.script else max(a for a, _ in clips) + 8
    tempos = reel.fit_tempos(clips, end)
    for (at, path), tempo in zip(clips, tempos):
        if tempo > 1.0:
            print(f"  {path.name}: sped up x{tempo} to finish before the next line", flush=True)
    voice = args.out / "narration.wav"
    subprocess.run(reel.mix_command(ffmpeg, clips, voice, None, args.music_volume, tempos), check=True)
    print(f"Voice track: {voice}")
    if args.video and args.video.is_file():
        final = args.out / "final_reel.mp4"
        if reel.has_audio(ffmpeg, args.video):
            cmd = reel.mix_command(ffmpeg, clips, final, args.video, args.music_volume, tempos)
        else:
            cmd = [ffmpeg, "-y", "-i", str(args.video), "-i", str(voice), "-map", "0:v", "-map", "1:a",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)]
        subprocess.run(cmd, check=True)
        print(f"Final reel: {final}")
    else:
        print("No video given: upload the exported reel as reels/reel.mp4 to get the mixed video.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
