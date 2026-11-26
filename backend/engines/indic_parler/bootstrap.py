"""Indic Parler-TTS venv probe + lazy bootstrap.

The runtime is Hugging Face's ``parler-tts`` package (Apache-2.0), which pins
``transformers==4.46.1`` — a conflict with the parent's ``transformers>=5.3``,
so it runs in its **own subprocess venv** like Confucius4 / dots.tts. The same
venv also serves Meta's MMS-TTS Malayalam (``VitsModel`` ships in that
transformers), so one install backs both engines.

Probe order (existing power-user installs win — zero migration):

    1. ``${OMNIVOICE_INDIC_PARLER_DIR}/.venv/`` — the user's clone-level venv.
    2. ``backend/engines/indic_parler/.venv/`` — this package's own venv.
    3. Bootstrap: ``uv venv`` then ``uv pip install -e <clone>``.

Opt-in (env-dir gated): never touched unless ``OMNIVOICE_INDIC_PARLER_DIR`` is
set, so it cannot affect the default install on any platform.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from engines._venv_probe import ProbeResult, log_safe, venv_can_import

logger = logging.getLogger("omnivoice.indic_parler.bootstrap")

#: Absolute path to the sidecar entrypoint.
INDIC_PARLER_SIDECAR_SCRIPT: Path = Path(__file__).parent / "main.py"

#: This package's owned venv (Probe 2).
_ENGINES_VENV_DIR: Path = Path(__file__).parent / ".venv"


def _uv_env() -> "dict[str, str] | None":
    """uv cache co-location for installs on a non-system volume (D:-drive /
    portable installs): without it uv stages every wheel on the system drive
    and cross-volume COPIES it into the venv. Canonical logic lives in
    services.sidecar_install.uv_subprocess_env (lazy import, like _locate_uv).
    """
    from services.sidecar_install import uv_subprocess_env
    return uv_subprocess_env(_ENGINES_VENV_DIR.parent.parent)

#: Env var pointing at the user's Indic Parler-TTS clone root.
_CLONE_DIR_ENV: str = "OMNIVOICE_INDIC_PARLER_DIR"

#: The package importable from the clone (verify against upstream).
_IMPORT_PROBE = "parler_tts"

_resolved_python: Optional[Path] = None

_UV_VENV_TIMEOUT_S = 120
_UV_PIP_INSTALL_TIMEOUT_S = 1800


def invalidate() -> None:
    """Clear the resolved-python cache. Tests call this between scenarios."""
    global _resolved_python
    _resolved_python = None


def _clone_dir() -> str | None:
    """Use the same launch-relative path semantics as the child sidecar."""
    raw = os.environ.get(_CLONE_DIR_ENV, "").strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else None


def is_indic_parler_installed() -> bool:
    """Cheap file-existence check for a usable venv (no subprocess spawn)."""
    return any(cand.is_file() for cand in _probe_paths())


def resolve_indic_parler_venv() -> Path:
    """Resolve the sidecar's Python interpreter (probe order in the docstring).
    Memoised. Raises :exc:`RuntimeError` if none can be located and bootstrap
    is unavailable."""
    global _resolved_python
    if _resolved_python is not None:
        return _resolved_python

    clone_dir = _clone_dir()

    # A candidate whose probe ran out of time (#1414): preferred over
    # bootstrapping or declaring the engine missing, but only after every
    # candidate has had its chance to prove itself outright.
    unproven: Optional[Path] = None

    if clone_dir:
        cand = _venv_python_path(Path(clone_dir) / ".venv")
        if cand.is_file():
            verdict = _venv_can_import(cand)
            if verdict == "yes":
                logger.info("Indic Parler-TTS venv resolved from %s: %s", _CLONE_DIR_ENV, cand)
                _resolved_python = cand
                return cand
            if verdict == "unproven":
                unproven = cand

    cand = _venv_python_path(_ENGINES_VENV_DIR)
    if cand.is_file():
        verdict = _venv_can_import(cand)
        if verdict == "yes":
            logger.info("Indic Parler-TTS venv resolved from engines path: %s", cand)
            _resolved_python = cand
            return cand
        if verdict == "unproven" and unproven is None:
            unproven = cand

    if unproven is not None:
        # Nothing proved itself, but something plausible is installed. Use
        # it: a venv that really is broken fails the sidecar handshake with
        # a real error, which beats reinstalling over the top of a working
        # install or telling the user their engine isn't there.
        logger.warning(
            "Indic Parler-TTS venv %s could not be verified in time; using it "
            "anyway rather than treating a slow import as a missing "
            "install (#1414).",
            log_safe(unproven),
        )
        _resolved_python = unproven
        return unproven

    if not clone_dir:
        raise RuntimeError(
            "Indic Parler-TTS is not installed. Set the "
            f"{_CLONE_DIR_ENV} environment variable to your Indic Parler-TTS clone "
            "(the parler-tts checkout), then restart "
            "VoiceStudio. See docs/engines/indic_parler-tts.md."
        )

    cand = _bootstrap_engines_venv(Path(clone_dir))
    _resolved_python = cand
    return cand


def _venv_python_path(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _probe_paths() -> list[Path]:
    out: list[Path] = []
    clone_dir = _clone_dir()
    if clone_dir:
        out.append(_venv_python_path(Path(clone_dir) / ".venv"))
    out.append(_venv_python_path(_ENGINES_VENV_DIR))
    return out


def _import_probe_code() -> str:
    """Probe snippet mirroring the sidecar's import semantics: the clone goes
    first on sys.path so an editable install and a plain checkout both work."""
    clone = _clone_dir()
    if clone:
        return f"import sys; sys.path.insert(0, {clone!r}); import {_IMPORT_PROBE}"
    return f"import {_IMPORT_PROBE}"


def _venv_can_import(python_path: Path) -> ProbeResult:
    """Spawn the candidate python and verify the engine imports.

    Tri-state — "yes" / "no" / "unproven". See ``engines._venv_probe``:
    a probe that runs out of time proves nothing, and treating that as
    "no" is what discarded working user installs (#1414).
    """
    return venv_can_import(
        python_path, _import_probe_code(), engine="indic_parler", logger=logger,
    )


def _locate_uv() -> Optional[str]:
    bundled = os.environ.get("OMNIVOICE_BUNDLED_UV")
    if bundled and Path(bundled).is_file():
        return bundled
    return shutil.which("uv")


def _bootstrap_engines_venv(clone_dir: Path) -> Path:
    """Create engines/indic_parler/.venv and install the user's clone."""
    uv = _locate_uv()
    if not uv:
        raise RuntimeError(
            "uv is required to bootstrap the Indic Parler-TTS venv but was not "
            "found on PATH (and OMNIVOICE_BUNDLED_UV was not set). Install uv "
            "from https://docs.astral.sh/uv/ and re-launch VoiceStudio."
        )

    logger.info(
        "Bootstrapping Indic Parler-TTS venv at %s from %s (several minutes on first "
        "launch)", _ENGINES_VENV_DIR, clone_dir,
    )
    try:
        subprocess.run(
            [uv, "venv", "--python", "3.10", str(_ENGINES_VENV_DIR)],
            check=True, timeout=_UV_VENV_TIMEOUT_S, capture_output=True,
            env=_uv_env(),
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"uv venv failed for Indic Parler-TTS bootstrap at {_ENGINES_VENV_DIR}: "
            f"{exc.stderr.decode('utf-8', errors='replace') if exc.stderr else exc}"
        ) from exc

    python_path = _venv_python_path(_ENGINES_VENV_DIR)
    requirements = clone_dir / "requirements.txt"
    try:
        if requirements.is_file():
            subprocess.run(
                [uv, "pip", "install", "--python", str(python_path),
                 "-r", str(requirements)],
                check=True, timeout=_UV_PIP_INSTALL_TIMEOUT_S, capture_output=True,
                env=_uv_env(),
            )
        # Upstream ships setup.py/pyproject.toml (package ``parler_tts``).
        if (clone_dir / "pyproject.toml").is_file() or (clone_dir / "setup.py").is_file():
            subprocess.run(
                [uv, "pip", "install", "--python", str(python_path), "-e", str(clone_dir)],
                check=True, timeout=_UV_PIP_INSTALL_TIMEOUT_S, capture_output=True,
                env=_uv_env(),
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "uv pip install failed during Indic Parler-TTS bootstrap "
            f"({clone_dir}): "
            f"{exc.stderr.decode('utf-8', errors='replace') if exc.stderr else exc}. "
            "See docs/engines/indic_parler-tts.md."
        ) from exc

    # Only a *proven* failure is fatal: a bootstrap that installed correctly
    # and is merely slow to import must not be thrown away after spending
    # minutes on the install (#1414).
    if _venv_can_import(python_path) == "no":
        raise RuntimeError(
            f"Indic Parler-TTS bootstrap completed but `import {_IMPORT_PROBE}` still "
            f"fails from {python_path}. Verify {clone_dir} is a valid clone. "
            "See docs/engines/indic_parler-tts.md."
        )

    logger.info("Indic Parler-TTS venv bootstrap successful: %s", python_path)
    return python_path


__all__ = [
    "INDIC_PARLER_SIDECAR_SCRIPT",
    "invalidate",
    "is_indic_parler_installed",
    "resolve_indic_parler_venv",
]
