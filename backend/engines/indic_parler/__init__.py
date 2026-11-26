"""Indic Parler-TTS and MMS-TTS Malayalam — free voices that need no reference clip.

Two engines share one sidecar and one venv (Hugging Face ``parler-tts``, which
pins ``transformers==4.46.1``):

* ``indic-parler-tts`` — AI4Bharat's Indic Parler-TTS (Apache-2.0 weights):
  21 languages including Malayalam. The voice is **described in words**
  ("Anjali speaks in a calm, clear voice…") — VoiceStudio's Voice Design
  details become that description. Malayalam has named speakers (Anjali,
  Anju, Harish) and emotion prompts.
* ``mms-tts-mal`` — Meta's MMS-TTS Malayalam (VITS, CC-BY-NC-4.0 weights): a
  small, fast, single-voice model. Non-commercial, so VoiceStudio asks for
  licence acceptance before it loads.

Neither clones a voice (``supports_cloning = False``).
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

logger = logging.getLogger("omnivoice.indic_parler")

#: Model card: "Assamese, Bengali, Bodo, Dogri, English, Gujarati, Hindi,
#: Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia,
#: Sanskrit, Santali, Sindhi, Tamil, Telugu, and Urdu."
PARLER_LANGUAGES: tuple[str, ...] = (
    "as", "bn", "brx", "doi", "en", "gu", "hi", "kn", "kok", "mai", "ml",
    "mni", "mr", "ne", "or", "sa", "sat", "sd", "ta", "te", "ur",
)

#: A description is required by the model. Used when no Voice Design details
#: are given: the model card's recommended speakers per language.
DEFAULT_SPEAKERS: dict[str, str] = {
    "ml": "Anjali", "ta": "Jaya", "te": "Prakash", "kn": "Suresh", "hi": "Rohit",
    "bn": "Arjun", "mr": "Sanjay", "gu": "Yash", "or": "Manas", "as": "Amit",
    "en": "Thoma",
}

_ENV_MODE = "VOICESTUDIO_INDIC_PARLER_MODE"


def default_description(language: object) -> str:
    """A neutral, clear-studio description for the requested language."""
    code = str(language or "ml").strip().lower()[:3]
    code = code if code in DEFAULT_SPEAKERS else "ml"
    speaker = DEFAULT_SPEAKERS[code]
    return (
        f"{speaker} speaks at a moderate pace with a clear, natural tone. "
        "The recording is of very high quality, with no background noise."
    )


class _SharedRuntime(SubprocessBackend):
    """Venv / sidecar resolution shared by both engines."""

    supports_cloning = False
    supports_num_step = False
    supports_guidance_scale = False
    gpu_compat = ("cuda", "mps", "cpu")
    _MODE = ""

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        from engines.indic_parler.bootstrap import (
            INDIC_PARLER_SIDECAR_SCRIPT,
            is_indic_parler_installed,
        )
        if not is_indic_parler_installed():
            return False, (
                "Indic Parler-TTS runtime not found. Install Indic Parler-TTS "
                "from Model Catalogue (it also runs MMS-TTS Malayalam), or set "
                "OMNIVOICE_INDIC_PARLER_DIR to a parler-tts checkout and restart "
                "VoiceStudio. See docs/engines/indic-parler-tts.md."
            )
        if not INDIC_PARLER_SIDECAR_SCRIPT.exists():
            return False, (
                f"Sidecar script missing at {INDIC_PARLER_SIDECAR_SCRIPT} "
                "— reinstall VoiceStudio."
            )
        return True, "ok"

    @classmethod
    def venv_python(cls):
        from engines.indic_parler.bootstrap import resolve_indic_parler_venv
        return resolve_indic_parler_venv()

    @classmethod
    def sidecar_python_hint(cls) -> "Path | None":
        from services.subprocess_backend import bootstrap_python_hint

        from engines.indic_parler import bootstrap
        return bootstrap_python_hint(bootstrap)

    @classmethod
    def sidecar_script(cls):
        from engines.indic_parler.bootstrap import INDIC_PARLER_SIDECAR_SCRIPT
        return INDIC_PARLER_SIDECAR_SCRIPT

    def extra_spawn_env(self) -> dict:
        env = dict(super().extra_spawn_env())
        env[_ENV_MODE] = self._MODE
        return env

    @property
    def recv_timeout_s(self) -> float:
        try:
            v = float(os.environ.get("OMNIVOICE_INDIC_PARLER_RECV_TIMEOUT_S", "900"))
        except (ValueError, TypeError):
            return 900.0
        if not math.isfinite(v):
            return 900.0
        return max(30.0, v)

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE


class IndicParlerBackend(_SharedRuntime):
    """AI4Bharat Indic Parler-TTS — voice described in words, 21 languages."""

    id = "indic-parler-tts"
    # ai4bharat/indic-parler-tts is gated (accept terms on the Hub + token).
    requires_hf_auth = True
    display_name = (
        "Indic Parler-TTS (AI4Bharat, Malayalam + 20 languages, voice from a "
        "description, Apache-2.0)"
    )
    supports_voice_design = True
    # Descript Audio Codec 44.1 kHz; re-read from the sidecar's frames.
    _DEFAULT_SAMPLE_RATE = 44100
    _MODE = "parler"

    @property
    def supported_languages(self) -> list[str]:
        return list(PARLER_LANGUAGES)

    def generate(self, text: str, **kw) -> "torch.Tensor":
        """kwargs honored: ``instruct`` / ``description`` (the voice, in
        words; a clear default speaker for the language when absent) and
        ``language``. ``ref_audio`` is ignored: this engine does not clone."""
        self._check_language(kw.get("language"))
        if kw.get("ref_audio"):
            logger.info("Indic Parler-TTS does not clone; ignoring the reference clip.")
        description = (kw.get("instruct") or kw.get("description") or "").strip()
        forwarded: dict = {
            "description": description or default_description(kw.get("language")),
        }
        if kw.get("language"):
            forwarded["language"] = str(kw["language"])
        return super().generate(text, **forwarded)


class MMSMalayalamBackend(_SharedRuntime):
    """Meta MMS-TTS Malayalam — one fixed voice, small and fast (CC-BY-NC)."""

    id = "mms-tts-mal"
    display_name = "MMS-TTS Malayalam (Meta, one voice, small and fast, CC-BY-NC)"
    supports_voice_design = False
    _DEFAULT_SAMPLE_RATE = 16000
    _MODE = "mms"

    @property
    def supported_languages(self) -> list[str]:
        return ["ml"]

    def generate(self, text: str, **kw) -> "torch.Tensor":
        self._check_language(kw.get("language"))
        if kw.get("ref_audio") or kw.get("instruct"):
            logger.info("MMS-TTS Malayalam has one fixed voice; ignoring voice inputs.")
        forwarded: dict = {"language": "ml"}
        return super().generate(text, **forwarded)


__all__ = [
    "DEFAULT_SPEAKERS", "IndicParlerBackend", "MMSMalayalamBackend",
    "PARLER_LANGUAGES", "default_description",
]
