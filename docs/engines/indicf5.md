# IndicF5 (opt-in engine) — Malayalam and Indian languages

> **Status: integrated, not yet run on real hardware by the maintainers.** The
> engine registration, dedicated-venv bootstrap, sidecar wire protocol and
> gating are unit-tested (`tests/test_indicf5_sidecar.py`); the upstream API
> was read from its README and source, not executed. It is opt-in and inert
> until you install it, so it cannot affect the default install.

[IndicF5](https://github.com/AI4Bharat/IndicF5) (AI4Bharat, IIT Madras) is an
F5-TTS voice-cloning model trained on 1,417 hours of Indian speech (Rasa,
IndicTTS, LIMMITS, IndicVoices-R).

- **11 languages:** Assamese, Bengali, Gujarati, Hindi, Kannada, **Malayalam**,
  Marathi, Odia, Punjabi, Tamil, Telugu. Any other language (including English)
  is rejected with the standard "can't speak" error.
- **Why use it:** the default engine speaks Malayalam too, but IndicF5 was
  trained only on Indian speech, so it keeps a native Malayalam accent and
  rhythm.
- **Voice cloning from a reference clip _and its transcript_.** F5-TTS aligns
  on both. VoiceStudio voice profiles store an automatic transcript, so this
  usually needs nothing from you; check that the transcript is in Malayalam
  script and matches the clip. A request without one fails with a clear
  message (upstream would otherwise download a Whisper model on its own).
- **Output:** 24 kHz mono.
- **Hardware:** CUDA, Apple Silicon (MPS) or CPU. CPU works but is slow.
  Not offered on Intel Macs (no PyTorch build).
- **Licence:** the weights are **MIT** (model card). The repository is
  **gated**, and the GitHub source has no LICENSE file; see
  [Rights](#rights).

It runs in its **own subprocess venv** because upstream pins
`transformers<4.50` and `numpy<=1.26.4`, which conflict with VoiceStudio.

## Before you install: accept the model terms

1. Sign in at <https://huggingface.co/ai4bharat/IndicF5> and accept the
   access conditions.
2. Create a read token at <https://huggingface.co/settings/tokens> and add it
   in **Settings → Credentials** (Hugging Face token).

Without both, the first synthesis fails with a message pointing here.

## One-click install

**Model Catalogue → IndicF5 → Install** fetches the pinned upstream source
(`13f7c4d`) into its own folder under VoiceStudio's data directory and builds a
Python 3.10 environment from upstream's `requirements.txt` plus the package
itself. Nothing it installs touches VoiceStudio or any other engine;
**Uninstall** removes only that folder. The ~1.4 GB of weights download on the
first synthesis, at the reviewed revision `fb652a9`.

## Manual install

```bash
git clone https://github.com/AI4Bharat/IndicF5.git
cd IndicF5
uv venv --python 3.10
# Windows with an NVIDIA GPU: install CUDA torch first, e.g.
# uv pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install -e .
```

Then point VoiceStudio at the clone and restart:

- **macOS/Linux:** `export OMNIVOICE_INDICF5_DIR=/path/to/IndicF5`
- **Windows (PowerShell):** `[Environment]::SetEnvironmentVariable("OMNIVOICE_INDICF5_DIR","C:\path\to\IndicF5","User")`

Or set it in **Settings → Environment**.

## Using it for Malayalam

1. Record or upload 5–15 seconds of clean Malayalam speech from the speaker
   you want, and save it as a voice profile. Check the transcript is the exact
   Malayalam words spoken.
2. Pick **IndicF5** as the engine and **Malayalam** as the language.
3. Type the text in Malayalam script (`എനിക്ക് മലയാളം ഇഷ്ടമാണ്`), not
   in English letters.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `OMNIVOICE_INDICF5_DIR` | managed install | IndicF5 checkout whose `.venv` runs the sidecar |
| `OMNIVOICE_INDICF5_MODEL` | `ai4bharat/IndicF5` | Another Hub repo or a local directory |
| `OMNIVOICE_INDICF5_REVISION` | — | Required 40-character commit for a custom Hub repo (the model runs remote code) |
| `OMNIVOICE_INDICF5_RECV_TIMEOUT_S` | `900` | Seconds a synthesis may run before the sidecar is stopped |

## Rights

- **Weights** (`ai4bharat/IndicF5` at `fb652a96e508091045890dae5bb99fe511facbcf`):
  MIT per the model card; gated with access conditions, which ask you to clone
  only voices you have permission to use.
- **Code** (`AI4Bharat/IndicF5` at `13f7c4d627cc10111aea8fe9c0039462cacacdc7`):
  no LICENSE file; it derives from F5-TTS (MIT code).
- **Vocoder:** Vocos (`charactr/vocos-mel-24khz`, MIT), fetched by upstream code.
- **Training data** terms are not yet reviewed.

The registry entry in `backend/config/model_licenses.json` is `unreviewed`:
these identifiers are upstream metadata, not verified permission for
commercial use.
