"""IndicF5 sidecar entry point.

Runs inside ``engines/indicf5/.venv`` (or the user's
``${OMNIVOICE_INDICF5_DIR}/.venv``), isolated from the OmniVoice parent —
upstream pins ``transformers<4.50`` and ``numpy<=1.26.4``.

Stdlib-only at import time; torch / transformers are imported lazily on the
first synthesize op so the ``ready`` frame fits inside the parent's 30 s spawn
handshake.

Wire protocol — length-prefixed JSON over stdin/stdout, byte-identical to
``backend/services/subprocess_backend.py``::

    [ 4-byte big-endian uint32 length ][ N bytes UTF-8 JSON ]

Op flow: ready → ping/pong → synthesize (→ progress, → audio) → shutdown.

Model API (upstream README): ``AutoModel.from_pretrained("ai4bharat/IndicF5",
trust_remote_code=True)`` then ``model(text, ref_audio_path=, ref_text=)`` →
numpy audio at 24 kHz. The remote code runs at a reviewed, immutable revision
only. The parent hands over a plain PCM WAV reference (#3688), which pydub
reads natively, so no ffmpeg is needed here.

Restrictions: NO imports from OmniVoice parent code. NO logging of os.environ.
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import sys
import traceback

MAX_FRAME_BYTES = 64 * 1024 * 1024

#: Vocos mel-24kHz vocoder rate used by upstream's example
#: (``sf.write(..., samplerate=24000)``).
INDICF5_SAMPLE_RATE = 24000

_DEFAULT_MODEL = "ai4bharat/IndicF5"
#: Reviewed immutable revision of the default repo. The model ships remote
#: code (``trust_remote_code=True``), so a branch would let new code run
#: unreviewed. Kept equal to the pin in backend/config/model_licenses.json by
#: tests/test_indicf5_sidecar.py.
_DEFAULT_REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"


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
        # A ping must stay cheap: never import torch here (#2837).
        torch = sys.modules.get("torch")
        if torch is None:
            return 0.0
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
    except Exception:
        pass
    return 0.0


def _model_source() -> tuple[str, str | None]:
    """(repo id or local directory, revision). A local directory is used as
    given; a Hub repo other than the default needs an explicit commit SHA in
    ``OMNIVOICE_INDICF5_REVISION`` because its remote code runs here."""
    model = os.environ.get("OMNIVOICE_INDICF5_MODEL", "").strip() or _DEFAULT_MODEL
    if os.path.isdir(model):
        return model, None
    revision = _DEFAULT_REVISION if model == _DEFAULT_MODEL else (
        os.environ.get("OMNIVOICE_INDICF5_REVISION", "").strip())
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(
            "A custom IndicF5 model needs OMNIVOICE_INDICF5_REVISION=<40-character "
            "commit SHA>; branches and tags are mutable and the model runs "
            "remote code. A local directory needs no revision."
        )
    return model, revision


def _pick_device(torch) -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass  # broken accelerator drivers must not block CPU loading
    return "cpu"


_model = None


def _load_model(stdout):
    global _model
    if _model is not None:
        return _model

    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 0})
    import torch
    from transformers import AutoModel

    source, revision = _model_source()
    kwargs: dict = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 30})
    try:
        model = AutoModel.from_pretrained(source, **kwargs)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if re.search(r"gated|401|403|restricted|access", text, re.IGNORECASE):
            raise RuntimeError(
                "IndicF5 weights are gated. Sign in at "
                "https://huggingface.co/ai4bharat/IndicF5, accept the terms, "
                "add your Hugging Face token in Settings, then try again. "
                f"({text})"
            ) from exc
        raise
    device = _pick_device(torch)
    to = getattr(model, "to", None)
    if callable(to) and device != "cpu":
        try:
            model = to(device)
        except Exception:
            model = to("cpu")  # an accelerator that cannot hold it: stay usable
    eval_ = getattr(model, "eval", None)
    if callable(eval_):
        model = eval_() or model
    _model = model
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _model


def _audio_to_pcm_b64(audio) -> tuple[str, int]:
    """Upstream returns int16 or float numpy audio; normalise to int16 PCM."""
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().to("cpu").numpy()
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    while arr.ndim > 1:
        # Downmix along the channel (shortest) axis, never time (#1328).
        arr = arr.mean(axis=int(np.argmin(arr.shape)))
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii"), int(arr.shape[0])


def _handle_synthesize(msg: dict, stdout) -> None:
    text = msg.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("synthesize: missing or non-string 'text'")
    ref_audio = msg.get("ref_audio")
    if not isinstance(ref_audio, str) or not ref_audio.strip():
        raise ValueError(
            "IndicF5 clones a voice: pass ref_audio with a path to a reference clip."
        )
    ref_text = msg.get("ref_text")
    if not isinstance(ref_text, str) or not ref_text.strip():
        # Upstream would otherwise download and run Whisper on its own.
        raise ValueError(
            "IndicF5 needs the transcript of the reference clip (ref_text)."
        )
    model = _load_model(stdout)
    audio = model(text, ref_audio_path=ref_audio, ref_text=ref_text.strip())
    sample_rate = int(getattr(model, "sample_rate", 0) or INDICF5_SAMPLE_RATE)
    pcm_b64, n_samples = _audio_to_pcm_b64(audio)
    _send(stdout, {
        "op": "audio",
        "audio_pcm_b64": pcm_b64,
        "sample_rate": sample_rate,
        "n_samples": n_samples,
    })


def main() -> int:
    stdin = sys.stdin.buffer
    # Frames go down a PRIVATE fd; fd 1 is pointed at stderr so library
    # prints (tqdm, upstream print()) cannot corrupt the frame stream (#1428).
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(_frame_fd, "wb")

    _send(stdout, {
        "op": "ready",
        "engine": "indicf5",
        "sample_rate": INDICF5_SAMPLE_RATE,
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
