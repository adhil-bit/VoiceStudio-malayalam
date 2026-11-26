# Indic Parler-TTS and MMS-TTS Malayalam (opt-in engines)

> **Status: integrated, not yet run on real hardware by the maintainers.** The
> engine registration, dedicated-venv bootstrap, sidecar wire protocol and
> gating are unit-tested (`tests/test_indic_parler_sidecar.py`); the upstream
> APIs were read from the model cards and source, not executed. Both are
> opt-in and inert until you install them.

Two free engines that speak Malayalam **without a reference clip**. They share
one install: Hugging Face's [parler-tts](https://github.com/huggingface/parler-tts)
runtime in its own Python 3.10 environment (it pins `transformers==4.46.1`,
which conflicts with VoiceStudio).

| | Indic Parler-TTS | MMS-TTS Malayalam |
|---|---|---|
| Engine id | `indic-parler-tts` | `mms-tts-mal` |
| Model | [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) | [`facebook/mms-tts-mal`](https://huggingface.co/facebook/mms-tts-mal) |
| Voice | **Described in words** (Voice Design details) | One fixed voice |
| Languages | 21: Assamese, Bengali, Bodo, Dogri, English, Gujarati, Hindi, Kannada, Konkani, Maithili, **Malayalam**, Manipuri, Marathi, Nepali, Odia, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu | Malayalam |
| Output | 44.1 kHz | 16 kHz |
| Weights | ~3.8 GB, gated, **Apache-2.0** | ~150 MB, **CC-BY-NC-4.0 (non-commercial)** |
| Clones a voice | No | No |

Both run on CUDA, Apple Silicon (MPS) or CPU, and are not offered on Intel
Macs (no PyTorch build). For cloning a Malayalam voice, use
[IndicF5](indicf5.md) or the default engine.

## Install

1. Indic Parler-TTS is gated: sign in at
   <https://huggingface.co/ai4bharat/indic-parler-tts>, accept the access
   conditions, and add a Hugging Face read token in **Settings → Credentials**.
   (MMS-TTS is not gated.)
2. **Model Catalogue → Indic Parler-TTS → Install.** This fetches the pinned
   parler-tts source (`d108732`) into its own folder and builds its
   environment. It also makes **MMS-TTS Malayalam** available; that engine has
   no separate install. **Uninstall** removes only that folder.
3. Weights download on the first synthesis, at reviewed revisions.

Manual install:

```bash
git clone https://github.com/huggingface/parler-tts.git
cd parler-tts
uv venv --python 3.10
# Windows with an NVIDIA GPU: install CUDA torch first, e.g.
# uv pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
export OMNIVOICE_INDIC_PARLER_DIR=$PWD   # or Settings → Environment
```

## Indic Parler-TTS: describing the voice

Pick **Indic Parler-TTS** and **Malayalam**, type Malayalam script, and write
the voice in **Voice Design** details. Name one of the Malayalam speakers the
model knows — **Anjali** or **Harish** (recommended), or **Anju** — and say
how they should sound:

> Harish speaks slowly in a deep, calm voice. The recording is very clear,
> with no background noise.

> Anjali speaks with a happy, energetic tone at a fast pace, close to the
> microphone.

The model card lists Malayalam among the languages with tested emotion
prompts. With no description, VoiceStudio uses a neutral one with Anjali.

## MMS-TTS Malayalam

Pick **MMS-TTS Malayalam** to get one plain Malayalam voice quickly; it
ignores voice profiles and Voice Design. Its weights are **non-commercial**
(CC-BY-NC-4.0), so VoiceStudio asks you to accept that licence before it first
loads.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `OMNIVOICE_INDIC_PARLER_DIR` | managed install | parler-tts checkout whose `.venv` runs both engines |
| `OMNIVOICE_INDIC_PARLER_RECV_TIMEOUT_S` | `900` | Seconds a synthesis may run before the sidecar is stopped |

## Rights

- `ai4bharat/indic-parler-tts` at `660bc53c64ae83150613264111708a0465211237`:
  Apache-2.0 per the model card; training datasets carry their own terms.
- `google/flan-t5-large` at `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
  (tokenizer only): Apache-2.0.
- `facebook/mms-tts-mal` at `893b8c6442d6a630896d1d3ac0f429094ddfae82`:
  CC-BY-NC-4.0, non-commercial.
- Runtime `huggingface/parler-tts` at `d108732`: Apache-2.0.

All registry entries in `backend/config/model_licenses.json` are
`unreviewed`: upstream metadata, not verified permission for commercial use.
