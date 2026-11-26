#!/usr/bin/env python3
"""Download free, openly licensed Malayalam reference voices for cloning.

Picks one clean 6–15 second clip per speaker from a public Malayalam speech
corpus on Hugging Face, with its exact transcript, and writes a folder you can
import as VoiceStudio voice profiles (New voice → upload the .wav, paste the
.txt transcript). Nothing runs unless you start it; it downloads only the
clips it keeps (through the Hugging Face dataset viewer API), never the whole
corpus.

Default source: IMaSC — ICFOSS Malayalam Speech Corpus (``thennal/IMaSC``),
8 speakers recorded in a studio, CC BY-SA 4.0. Other corpora with ``audio``,
text and speaker columns work through the options below, for example the SMC
Malayalam Speech Corpus (``thennal/msc``, CC BY-SA 4.0, columns
``transcript``/``speaker_id``).

    python scripts/malayalam_reference_voices.py --out malayalam-voices
    python scripts/malayalam_reference_voices.py --dataset thennal/msc \\
        --text-column transcript --speaker-column speaker_id --per-speaker 1 \\
        --max-speakers 10 --licence "CC BY-SA 4.0" \\
        --credit "Swathanthra Malayalam Computing — Malayalam Speech Corpus"

CC BY-SA means: credit the corpus (ATTRIBUTION.md is written for you) and
share changes to the clips under the same licence. The people in these
recordings agreed to an open corpus, not to have their voices imitated in
anything you publish; use the voices respectfully and say that speech is
synthetic.

Standard library only, so it runs with any Python 3.9+.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

VIEWER = "https://datasets-server.huggingface.co"
DEFAULT_DATASET = "thennal/IMaSC"
DEFAULT_LICENCE = "CC BY-SA 4.0"
DEFAULT_CREDIT = (
    "IMaSC — ICFOSS Malayalam Speech Corpus (Gopinath, D K, Nair, K S, G; "
    "ICFOSS, 2022), https://huggingface.co/datasets/thennal/IMaSC"
)
MIN_SECONDS = 6.0
MAX_SECONDS = 15.0
PAGE = 100
_MALAYALAM = re.compile(r"[ഀ-ൿ]")
USER_AGENT = "VoiceStudio-malayalam-reference-voices/1"


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - https only
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    return f"{VIEWER}/rows?{query}"


def audio_src(cell) -> str | None:
    """The viewer gives an audio cell as ``[{"src": url, "type": mime}]``."""
    items = cell if isinstance(cell, list) else [cell]
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("src"), str):
            if item.get("type", "audio/wav") in ("audio/wav", "audio/x-wav", "audio/wave"):
                return item["src"]
    return None


def wav_seconds(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def good_text(text: object) -> bool:
    """A usable transcript: Malayalam script, a sentence, no markup."""
    return (
        isinstance(text, str) and 20 <= len(text.strip()) <= 400
        and bool(_MALAYALAM.search(text)) and "<" not in text
    )


def safe_name(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
    return name[:40] or "speaker"


def collect(dataset: str, config: str, split: str, text_col: str, speaker_col: str,
            per_speaker: int, max_speakers: int, max_rows: int, fetch=None):
    """Yield (speaker, index, text, wav_bytes, seconds) for accepted clips."""
    fetch = fetch or _get
    counts: dict[str, int] = {}
    offset = 0
    while offset < max_rows:
        page = json.loads(fetch(rows_url(dataset, config, split, offset, PAGE)))
        rows = page.get("rows") or []
        if not rows:
            return
        for entry in rows:
            row = entry.get("row") or {}
            speaker = str(row.get(speaker_col, "speaker"))
            if counts.get(speaker, 0) >= per_speaker:
                continue
            if speaker not in counts and len(counts) >= max_speakers:
                continue
            text = row.get(text_col)
            src = audio_src(row.get("audio"))
            if not good_text(text) or not src:
                continue
            data = fetch(src)
            seconds = wav_seconds(data)
            if seconds is None or not MIN_SECONDS <= seconds <= MAX_SECONDS:
                continue
            counts[speaker] = counts.get(speaker, 0) + 1
            yield speaker, counts[speaker], text.strip(), data, seconds
        if (len(counts) >= max_speakers
                and all(n >= per_speaker for n in counts.values())):
            return
        offset += PAGE


def attribution(dataset: str, licence: str, credit: str, clips: list[dict]) -> str:
    lines = [
        "# Malayalam reference voices — attribution",
        "",
        f"Source: {credit}",
        f"Dataset: https://huggingface.co/datasets/{dataset}",
        f"Licence: {licence}",
        "",
        "The clips below are unmodified excerpts. If you share them, or audio",
        "you changed from them, keep this credit and the same licence.",
        "",
        "| File | Speaker | Seconds |",
        "|---|---|---|",
    ]
    lines += [f"| {c['file']} | {c['speaker']} | {c['seconds']:.1f} |" for c in clips]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("malayalam-voices"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--speaker-column", default="speaker")
    parser.add_argument("--per-speaker", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=5000,
                        help="stop after scanning this many rows")
    parser.add_argument("--licence", default=DEFAULT_LICENCE)
    parser.add_argument("--credit", default=DEFAULT_CREDIT)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    for speaker, n, text, data, seconds in collect(
        args.dataset, args.config, args.split, args.text_column,
        args.speaker_column, args.per_speaker, args.max_speakers, args.max_rows,
    ):
        stem = f"{safe_name(speaker)}-{n}"
        (args.out / f"{stem}.wav").write_bytes(data)
        (args.out / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        clips.append({"file": f"{stem}.wav", "speaker": speaker,
                      "seconds": round(seconds, 2), "transcript": text})
        print(f"saved {stem}.wav ({seconds:.1f}s)")
    if not clips:
        print("No suitable clips found. Check the dataset and column names.",
              file=sys.stderr)
        return 1
    (args.out / "voices.json").write_text(
        json.dumps({"dataset": args.dataset, "licence": args.licence,
                    "credit": args.credit, "clips": clips},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "ATTRIBUTION.md").write_text(
        attribution(args.dataset, args.licence, args.credit, clips), encoding="utf-8")
    print(f"\n{len(clips)} voices in {args.out}. In VoiceStudio: New voice → "
          "upload a .wav and paste its .txt as the transcript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
