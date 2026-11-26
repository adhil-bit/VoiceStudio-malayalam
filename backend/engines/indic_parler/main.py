"""Indic Parler-TTS / MMS-TTS Malayalam sidecar entry point.

Runs inside the parler-tts venv (``transformers==4.46.1``), isolated from the
OmniVoice parent. ``VOICESTUDIO_INDIC_PARLER_MODE`` (set by the parent per
engine) selects ``parler`` (AI4Bharat Indic Parler-TTS) or ``mms`` (Meta
MMS-TTS Malayalam, ``VitsModel``).

Stdlib-only at import time; torch / transformers load on the first synthesize
op so the ``ready`` frame fits inside the parent's 30 s spawn handshake.

Wire protocol — length-prefixed JSON over stdin/stdout, byte-identical to
``backend/services/subprocess_backend.py``::

    [ 4-byte big-endian uint32 length ][ N bytes UTF-8 JSON ]

Every Hub download uses a reviewed, immutable revision.

Restrictions: NO imports from OmniVoice parent code. NO logging of os.environ.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import traceback

MAX_FRAME_BYTES = 64 * 1024 * 1024

PARLER_MODEL = "ai4bharat/indic-parler-tts"
#: Reviewed revisions, kept equal to backend/config/model_licenses.json by
#: tests/test_indic_parler_sidecar.py.
PARLER_REVISION = "660bc53c64ae83150613264111708a0465211237"
#: The description encoder's tokenizer the model config names.
DESCRIPTION_TOKENIZER = "google/flan-t5-large"
DESCRIPTION_TOKENIZER_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
MMS_MODEL = "facebook/mms-tts-mal"
MMS_REVISION = "893b8c6442d6a630896d1d3ac0f429094ddfae82"

SAMPLE_RATES = {"parler": 44100, "mms": 16000}


def _mode() -> str:
    mode = os.environ.get("VOICESTUDIO_INDIC_PARLER_MODE", "").strip().lower()
    return mode if mode in SAMPLE_RATES else "parler"


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!I", len(body)))
    stream.write(body)
    stream.flush()


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None  # EOF
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {n}")
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise IOError("short read")
        body.extend(chunk)
    return json.loads(bytes(body).decode("utf-8"))


def _measure_vram_mb() -> float:
    try:
        torch = sys.modules.get("torch")  # never import torch for a ping (#2837)
        if torch is None:
            return 0.0
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
    except Exception:
        pass
    return 0.0


def _pick_device(torch) -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _gated_hint(exc: Exception, repo: str) -> Exception:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if any(k in lowered for k in ("gated", "401", "403", "restricted", "access to model")):
        return RuntimeError(
            f"{repo} is gated. Sign in at https://huggingface.co/{repo}, accept "
            "the terms, add your Hugging Face token in Settings, then try "
            f"again. ({text})"
        )
    return exc


_loaded: dict = {}


def _load(stdout) -> dict:
    """Load the selected model once. Returns a dict of the parts."""
    if _loaded:
        return _loaded
    mode = _mode()
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 0})
    import torch

    device = _pick_device(torch)
    try:
        if mode == "mms":
            from transformers import AutoTokenizer, VitsModel

            model = VitsModel.from_pretrained(MMS_MODEL, revision=MMS_REVISION)
            tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL, revision=MMS_REVISION)
            parts = {"model": model, "tokenizer": tokenizer}
        else:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            model = ParlerTTSForConditionalGeneration.from_pretrained(
                PARLER_MODEL, revision=PARLER_REVISION,
            )
            tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL, revision=PARLER_REVISION)
            _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 60})
            name = getattr(getattr(model.config, "text_encoder", None), "_name_or_path", "")
            if name and name != DESCRIPTION_TOKENIZER:
                # Only the reviewed encoder is fetched; anything else must be
                # pinned before it can run here.
                raise RuntimeError(
                    f"Indic Parler-TTS names an unreviewed description encoder {name!r}."
                )
            description_tokenizer = AutoTokenizer.from_pretrained(
                DESCRIPTION_TOKENIZER, revision=DESCRIPTION_TOKENIZER_REVISION,
            )
            parts = {
                "model": model, "tokenizer": tokenizer,
                "description_tokenizer": description_tokenizer,
            }
    except Exception as exc:
        raise _gated_hint(exc, MMS_MODEL if mode == "mms" else PARLER_MODEL) from exc
    model = parts["model"]
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception:
            device = "cpu"  # an accelerator that cannot hold it: stay usable
            model = model.to("cpu")
    parts["model"] = model.eval()
    parts["device"] = device
    parts["torch"] = torch
    parts["mode"] = mode
    _loaded.update(parts)
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _loaded


def _sample_rate(parts: dict) -> int:
    model = parts.get("model")
    rate = getattr(getattr(model, "config", None), "sampling_rate", None)
    try:
        return int(rate) if rate else SAMPLE_RATES[parts.get("mode") or _mode()]
    except (TypeError, ValueError):
        return SAMPLE_RATES[parts.get("mode") or _mode()]


def _audio_to_pcm_b64(audio) -> tuple[str, int]:
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().to("cpu").float().numpy()
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    while arr.ndim > 1:
        # Downmix along the channel (shortest) axis, never time (#1328).
        arr = arr.mean(axis=int(np.argmin(arr.shape)))
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii"), int(arr.shape[0])


def _synthesize(parts: dict, text: str, description: str):
    torch = parts["torch"]
    model, device = parts["model"], parts["device"]
    with torch.no_grad():
        if parts["mode"] == "mms":
            inputs = parts["tokenizer"](text, return_tensors="pt").to(device)
            return model(**inputs).waveform
        desc = parts["description_tokenizer"](description, return_tensors="pt").to(device)
        prompt = parts["tokenizer"](text, return_tensors="pt").to(device)
        return model.generate(
            input_ids=desc.input_ids, attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
        )


def _handle_synthesize(msg: dict, stdout) -> None:
    text = msg.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("synthesize: missing or non-string 'text'")
    description = msg.get("description")
    if _mode() == "parler" and (not isinstance(description, str) or not description.strip()):
        raise ValueError("Indic Parler-TTS needs a voice description.")
    parts = _load(stdout)
    audio = _synthesize(parts, text, (description or "").strip())
    pcm_b64, n_samples = _audio_to_pcm_b64(audio)
    _send(stdout, {
        "op": "audio",
        "audio_pcm_b64": pcm_b64,
        "sample_rate": _sample_rate(parts),
        "n_samples": n_samples,
    })


def main() -> int:
    stdin = sys.stdin.buffer
    # Frames go down a PRIVATE fd; fd 1 is pointed at stderr so library
    # prints cannot corrupt the frame stream (#1428).
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(_frame_fd, "wb")

    mode = _mode()
    _send(stdout, {
        "op": "ready",
        "engine": "mms-tts-mal" if mode == "mms" else "indic-parler-tts",
        "sample_rate": SAMPLE_RATES[mode],
    })

    while True:
        try:
            msg = _recv(stdin)
        except Exception as exc:
            _send(stdout, {
                "op": "error", "stage": "recv",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            return 1
        if msg is None:
            return 0

        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "ping":
                _send(stdout, {"op": "pong", "vram_mb": _measure_vram_mb()})
            elif op == "synthesize":
                _handle_synthesize(msg, stdout)
            elif op == "shutdown":
                return 0
            else:
                _send(stdout, {"op": "error", "stage": "dispatch",
                               "message": f"unknown op: {op!r}"})
        except Exception as exc:
            _send(stdout, {
                "op": "error", "stage": op or "unknown",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    sys.exit(main())
