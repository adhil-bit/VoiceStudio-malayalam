#!/usr/bin/env python3
"""Narrate a Malayalam reel with your running VoiceStudio and mix it into the video.

VoiceStudio must be running on this computer (the desktop app, or
``python backend/main.py``). Each script line is generated with the engine you
choose, placed at its start time, and mixed over the video's music.

    # Indic Parler-TTS: the voice is described in words (no clip needed)
    python scripts/malayalam_reel.py --video reel.mp4

    # IndicF5: clone a Malayalam clip (and its transcript)
    python scripts/malayalam_reel.py --video reel.mp4 --engine indicf5 \\
        --ref-audio malayalam-voices/Joji-1.wav --ref-text-file malayalam-voices/Joji-1.txt

    # A saved VoiceStudio voice
    python scripts/malayalam_reel.py --video reel.mp4 --engine indicf5 --profile-id ab12cd34

Without --video it writes only the narration track. The built-in script is the
"Spotlight Effect" reel (scene starts 0, 5, 13, 20, 30, 36 s); pass
``--script my.json`` (a list of {"at": seconds, "text": "..."}) for another.

Needs ffmpeg on PATH (VoiceStudio installs one; any ffmpeg works) and
Python 3.9+. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SPOTLIGHT_REEL = [
    {"at": 0.2, "text": "ഒരു ചെറിയ തെറ്റ് പറ്റുമ്പോൾ… എല്ലാവരും നിന്നെ മാത്രം നോക്കുകയാണെന്ന് തോന്നിയിട്ടുണ്ടോ?"},
    {"at": 5.2, "text": "കോളേജ് ഫംഗ്ഷനിൽ പാടുമ്പോൾ മീരയുടെ ശബ്ദം ഒന്ന് പതറി."},
    {"at": 13.2, "text": "അന്ന് രാത്രി മുഴുവൻ അവൾ ഉറങ്ങിയില്ല. എല്ലാവരും കളിയാക്കും എന്ന് അവൾ ഉറപ്പിച്ചു."},
    {"at": 20.2, "text": "പിറ്റേന്ന് അവൾ കൂട്ടുകാരിയോട് ചോദിച്ചു. അവൾ ചിരിച്ചു: ഏത് പതർച്ച? നീ നന്നായി പാടിയല്ലോ!"},
    {"at": 30.2, "text": "ഇതാണ് സ്പോട്ട്‌ലൈറ്റ് ഇഫക്റ്റ്. നമ്മൾ വിചാരിക്കുന്നതിന്റെ പകുതി പോലും ആളുകൾ നമ്മളെ ശ്രദ്ധിക്കുന്നില്ല."},
    {"at": 36.2, "text": "അതുകൊണ്ട്… പേടിക്കാതെ മുന്നോട്ട് പോകൂ."},
]

DEFAULT_DESCRIPTION = (
    "Anjali speaks in a warm, gentle storytelling tone at a moderate pace, with "
    "slight pauses between sentences. The recording is very clear, close to the "
    "microphone, with no background noise."
)


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Encode form fields and files as multipart/form-data."""
    boundary = uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        if value is None:
            continue
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode()
        out += str(value).encode("utf-8") + b"\r\n"
    for name, path in files.items():
        data = Path(path).read_bytes()
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{Path(path).name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def generate(server: str, text: str, args, api_key: str | None, timeout: float) -> bytes:
    fields = {
        "text": text, "language": "ml", "engine": args.engine,
        "speed": args.speed, "effect_preset": "broadcast",
    }
    files = {}
    if args.profile_id:
        fields["profile_id"] = args.profile_id
    elif args.engine == "indic-parler-tts":
        fields["instruct"] = args.description
    elif args.ref_audio:
        files["ref_audio"] = args.ref_audio
        fields["ref_text"] = args.ref_text
    body, ctype = multipart(fields, files)
    req = urllib.request.Request(server.rstrip("/") + "/generate", data=body, method="POST",
                                 headers={"Content-Type": ctype})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:  # nosec B310 - local server
            return res.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"VoiceStudio refused the line ({exc.code}): {detail}") from None
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Can't reach VoiceStudio at {server} ({exc.reason}). Start the app first, "
            "or pass --server with its address."
        ) from None


def mix_command(ffmpeg: str, clips: list[tuple[float, Path]], out: Path,
                video: Path | None, music_volume: float) -> list[str]:
    """One ffmpeg call: each clip delayed to its start, mixed, and (with a
    video) laid over the video's own audio, which is turned down."""
    cmd = [ffmpeg, "-y"]
    if video:
        cmd += ["-i", str(video)]
    for _, path in clips:
        cmd += ["-i", str(path)]
    first = 1 if video else 0
    parts, labels = [], []
    for i, (at, _) in enumerate(clips):
        ms = max(0, int(round(at * 1000)))
        parts.append(f"[{first + i}:a]aresample=48000,adelay={ms}|{ms}[v{i}]")
        labels.append(f"[v{i}]")
    parts.append("".join(labels) + f"amix=inputs={len(clips)}:normalize=0[voice]")
    if video:
        parts.append(f"[0:a]volume={music_volume}[music]")
        parts.append("[music][voice]amix=inputs=2:duration=first:normalize=0[a]")
        cmd += ["-filter_complex", ";".join(parts), "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)]
    else:
        cmd += ["-filter_complex", ";".join(parts), "-map", "[voice]", str(out)]
    return cmd


def has_audio(ffmpeg: str, video: Path) -> bool:
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    return "Audio:" in probe.stderr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--video", type=Path, help="the reel exported as MP4 (optional)")
    p.add_argument("--out", type=Path, help="output file (default: final_reel.mp4 or narration.wav)")
    p.add_argument("--script", type=Path, help='JSON list of {"at": seconds, "text": "..."}')
    p.add_argument("--engine", default="indic-parler-tts",
                   help="indic-parler-tts (default), indicf5, omnivoice, mms-tts-mal …")
    p.add_argument("--description", default=DEFAULT_DESCRIPTION,
                   help="Indic Parler-TTS voice description (Anjali, Anju or Harish for Malayalam)")
    p.add_argument("--profile-id", help="a saved VoiceStudio voice to use")
    p.add_argument("--ref-audio", help="reference clip to clone (IndicF5 / OmniVoice)")
    p.add_argument("--ref-text", default="", help="exact transcript of the reference clip")
    p.add_argument("--ref-text-file", type=Path, help="read the transcript from this file")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--music-volume", type=float, default=0.3, help="0–1, how loud the reel's music stays")
    p.add_argument("--server", default=os.environ.get("VOICESTUDIO_URL", "http://127.0.0.1:3900"))
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--keep", type=Path, default=Path("reel-voice"), help="folder for the line clips")
    args = p.parse_args(argv)

    if args.ref_text_file:
        args.ref_text = args.ref_text_file.read_text("utf-8").strip()
    if args.engine == "indicf5" and not args.profile_id and not (args.ref_audio and args.ref_text):
        p.error("indicf5 needs --profile-id, or --ref-audio with --ref-text/--ref-text-file")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        p.error("ffmpeg not found on PATH")
    if args.video and not args.video.is_file():
        p.error(f"video not found: {args.video}")
    lines = json.loads(args.script.read_text("utf-8")) if args.script else SPOTLIGHT_REEL
    out = args.out or Path("final_reel.mp4" if args.video else "narration.wav")
    api_key = os.environ.get("OMNIVOICE_API_KEY")

    args.keep.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[float, Path]] = []
    for n, line in enumerate(lines, 1):
        print(f"[{n}/{len(lines)}] {line['at']:>5.1f}s  {line['text']}")
        wav = generate(args.server, line["text"], args, api_key, args.timeout)
        path = args.keep / f"line-{n:02d}.wav"
        path.write_bytes(wav)
        clips.append((float(line["at"]), path))

    video = args.video
    if video and not has_audio(ffmpeg, video):
        # A silent export: lay the voice on alone.
        silent = args.keep / "_silence.wav"
        subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "1", str(silent)],
                       check=True, capture_output=True)
        cmd = mix_command(ffmpeg, clips, out, None, args.music_volume)
        voice = args.keep / "_voice.wav"
        cmd[-1] = str(voice)
        subprocess.run(cmd, check=True, capture_output=True)
        cmd = [ffmpeg, "-y", "-i", str(video), "-i", str(voice), "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    else:
        cmd = mix_command(ffmpeg, clips, out, video, args.music_volume)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        print(result.stderr[-2000:], file=sys.stderr)
        return result.returncode
    print(f"\nDone: {out}  (line clips in {args.keep}/ — redo any line by running again)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
