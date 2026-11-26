"""IndicF5 sidecar package — Malayalam and 10 other Indian languages.

IndicF5 (AI4Bharat, https://github.com/AI4Bharat/IndicF5) is an F5-TTS
flow-matching voice-cloning model trained on 1,417 hours of Indian speech
(Rasa, IndicTTS, LIMMITS, IndicVoices-R). It speaks Assamese, Bengali,
Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu
with the accent of the reference clip, which is what users want when the
default engine's Malayalam sounds foreign.

It clones from a reference clip **and its transcript** (F5-TTS conditions on
both). VoiceStudio voice profiles already store an auto-transcribed
``ref_text``, so the usual path just works; a call without one fails with a
clear message instead of upstream silently downloading a Whisper model.

Like Confucius4 / dots.tts it runs in its **own subprocess venv** because it
pins ``transformers<4.50`` / ``numpy<=1.26.4``. It is **opt-in** — selected in
the engine picker after a one-click install (or ``OMNIVOICE_INDICF5_DIR``
pointing at a clone) — so it can never become a broken default.

Three entry points: ``IndicF5Backend`` (this module), ``main.py`` (the sidecar,
runs under the IndicF5 venv — never imported by the parent), and
``bootstrap.py`` (venv probe + lazy bootstrap).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

from services.subprocess_backend import SubprocessBackend

if TYPE_CHECKING:
    import torch  # noqa: F401

logger = logging.getLogger("omnivoice.indicf5")

#: Upstream model card: "IndicF5 supports 11 Indian languages".
INDICF5_LANGUAGES: tuple[str, ...] = (
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te",
)


class IndicF5Backend(SubprocessBackend):
    """IndicF5 (AI4Bharat) — F5-TTS voice cloning for 11 Indian languages."""

    id = "indicf5"
    # ai4bharat/IndicF5 is gated (accept terms on the Hub, then a token).
    requires_hf_auth = True
    display_name = (
        "IndicF5 (AI4Bharat, Malayalam + 10 Indian languages, voice cloning, MIT)"
    )
    supports_voice_design = False  # timbre comes from a reference clip
    # The sidecar reads text/ref_audio/ref_text/language only.
    supports_num_step = False
    supports_guidance_scale = False
    # Vocos mel-24kHz vocoder; re-read from the sidecar's ready/audio frames.
    _DEFAULT_SAMPLE_RATE = 24000
    gpu_compat = ("cuda", "mps", "cpu")

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        # Verify the venv on disk only — never import the engine here
        # (separate interpreter).
        from engines.indicf5.bootstrap import (
            INDICF5_SIDECAR_SCRIPT,
            is_indicf5_installed,
        )
        if not is_indicf5_installed():
            return False, (
                "IndicF5 venv not found. Install it from Model Catalogue, or set "
                "OMNIVOICE_INDICF5_DIR to your IndicF5 clone (the directory "
                "containing setup.py) and restart VoiceStudio. The weights are "
                "gated: accept the terms at huggingface.co/ai4bharat/IndicF5 and "
                "add your Hugging Face token in Settings. See "
                "docs/engines/indicf5.md."
            )
        if not INDICF5_SIDECAR_SCRIPT.exists():
            return False, (
                f"IndicF5 sidecar script missing at {INDICF5_SIDECAR_SCRIPT} "
                "— reinstall VoiceStudio."
            )
        return True, "ok"

    @classmethod
    def venv_python(cls):
        from engines.indicf5.bootstrap import resolve_indicf5_venv
        return resolve_indicf5_venv()

    @classmethod
    def sidecar_python_hint(cls) -> "Path | None":
        """Expose only the bootstrap-selected interpreter for routing metadata."""
        from services.subprocess_backend import bootstrap_python_hint

        from engines.indicf5 import bootstrap
        return bootstrap_python_hint(bootstrap)

    @classmethod
    def sidecar_script(cls):
        from engines.indicf5.bootstrap import INDICF5_SIDECAR_SCRIPT
        return INDICF5_SIDECAR_SCRIPT

    @property
    def recv_timeout_s(self) -> float:
        """Receive timeout for the sidecar. The first synthesis downloads the
        weights and CPU flow matching is slow, so the 60 s default is too short.
        ``OMNIVOICE_INDICF5_RECV_TIMEOUT_S`` tunes it."""
        try:
            v = float(os.environ.get("OMNIVOICE_INDICF5_RECV_TIMEOUT_S", "900"))
        except (ValueError, TypeError):
            return 900.0
        if not math.isfinite(v):
            return 900.0
        return max(30.0, v)

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        # Declared honestly so the base-class ``_check_language`` (#2104)
        # rejects e.g. English instead of producing an approximation.
        return list(INDICF5_LANGUAGES)

    def generate(self, text: str, **kw) -> "torch.Tensor":
        """Synthesize one utterance through the IndicF5 sidecar.

        kwargs honored:
          * ``ref_audio`` — reference clip path. **Required.**
          * ``ref_text`` — transcript of the reference clip. **Required**:
            F5-TTS aligns on it; without it upstream would download and run a
            Whisper model on its own.
          * ``language`` — validated against :attr:`supported_languages`; the
            model itself infers language from the script.

        Returns a tensor of shape (1, n_samples) at :attr:`sample_rate`.
        """
        self._check_language(kw.get("language"))
        ref_audio = kw.get("ref_audio")
        if not ref_audio:
            raise RuntimeError(
                "IndicF5 clones a voice: choose a voice profile or pass "
                "ref_audio= with a path to a reference clip in the target "
                "language."
            )
        ref_text = kw.get("ref_text")
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise RuntimeError(
                "IndicF5 needs the transcript of the reference clip. Add the "
                "words spoken in the clip to the voice profile (VoiceStudio "
                "fills this in automatically when it can) and try again."
            )
        forwarded: dict = {
            "ref_audio": os.path.abspath(os.fspath(ref_audio)),
            "ref_text": ref_text.strip(),
        }
        language = kw.get("language")
        if language:
            forwarded["language"] = str(language)
        return super().generate(text, **forwarded)


__all__ = ["INDICF5_LANGUAGES", "IndicF5Backend"]
