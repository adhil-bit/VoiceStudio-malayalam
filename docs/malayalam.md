# Malayalam in VoiceStudio

Everything free for Malayalam speech in one place: which engine to use, where
to get reference voices to clone, and open datasets for training.

## Engines

| Engine | How you choose the voice | Quality | Licence of weights | Guide |
|---|---|---|---|---|
| VoiceStudio (OmniVoice) — default | Clone any clip | Good; accent follows the clip | CC-BY-NC (non-commercial) | [omnivoice](engines/omnivoice.md) |
| **IndicF5** | Clone a clip **with its transcript** | Most natural Malayalam | MIT (gated) | [indicf5](engines/indicf5.md) |
| **Indic Parler-TTS** | **Describe** the voice in words (Anjali, Harish, Anju) | Natural; emotion prompts | Apache-2.0 (gated) | [indic-parler-tts](engines/indic-parler-tts.md) |
| **MMS-TTS Malayalam** | One fixed voice | Plain, fast, small | CC-BY-NC (non-commercial) | [indic-parler-tts](engines/indic-parler-tts.md#mms-tts-malayalam) |

Tips that apply to every engine:

- Choose **Malayalam** as the language, not Auto.
- Type Malayalam script (`എനിക്ക് മലയാളം ഇഷ്ടമാണ്`), not English letters.
- To clone, use a clean 5–15 second clip of a **Malayalam** speaker. An
  English clip gives Malayalam with an English accent.

Paid cloud services such as Sarvam Bulbul or Voicemaker are not included:
they need an account and API key and do not run on your computer.

## Free reference voices to clone

`scripts/malayalam_reference_voices.py` downloads one clean clip per speaker,
with its exact transcript, from an open Malayalam corpus:

```bash
python scripts/malayalam_reference_voices.py --out malayalam-voices
```

It needs only Python 3.9+ and internet access to Hugging Face, and downloads
only the clips it keeps (8 by default, a few MB). The default corpus is IMaSC
(8 studio speakers, CC BY-SA 4.0). For more variety, use the crowdsourced SMC
corpus:

```bash
python scripts/malayalam_reference_voices.py --dataset thennal/msc \
  --text-column transcript --speaker-column speaker_id --max-speakers 10 \
  --credit "Swathanthra Malayalam Computing — Malayalam Speech Corpus"
```

Then in VoiceStudio: **New voice → upload a `.wav` → paste the matching
`.txt` as the transcript**. The folder also has `ATTRIBUTION.md` and
`voices.json`.

CC BY-SA asks you to credit the corpus and to share changed clips under the
same licence. The speakers agreed to an open research corpus, not to having
their voices imitated in what you publish: use them respectfully and label the
result as synthetic speech.

## Open Malayalam speech datasets

For fine-tuning or evaluating a model. Check each licence before commercial
use; corpora are downloaded from their own pages, not by VoiceStudio.

| Dataset | What it is | Size | Licence | Where |
|---|---|---|---|---|
| IMaSC — ICFOSS Malayalam Speech Corpus | Studio TTS recordings, 8 speakers, Wikipedia text | ~50 h, 34,473 clips, 16 kHz | CC BY-SA 4.0 | [thennal/IMaSC](https://huggingface.co/datasets/thennal/IMaSC) · [paper](https://arxiv.org/abs/2211.12796) |
| SMC Malayalam Speech Corpus (MSC) | Crowdsourced reads, many speakers, reviewed | ~1–10k clips | CC BY-SA 4.0 | [thennal/msc](https://huggingface.co/datasets/thennal/msc) · [announcement](https://blog.smc.org.in/malayalam-speech-corpus) |
| OpenSLR 63 | Google crowdsourced high-quality multi-speaker | ~4k clips | CC BY-SA 4.0 | [openslr.org/63](https://www.openslr.org/63/) · [HF mirror](https://huggingface.co/datasets/vrclc/openslr63) |
| GMaSC — GEC Barton Hill | Malayalam-accented English, studio | 2,000 clips, 48 kHz | CC BY-SA 4.0 | [thennal/GMaSC](https://huggingface.co/datasets/thennal/GMaSC) |
| IndicVoices-R | Enhanced natural speech, 22 Indian languages incl. Malayalam | 1,704 h total, 10k+ speakers | CC BY 4.0 | [AI4Bharat/IndicVoices-R](https://github.com/AI4Bharat/IndicVoices-R) |
| Indic TTS (IIT Madras) — Malayalam | Studio TTS, 1 male + 1 female | ~8,600 clips | Custom licence (read it first) | [thennal/indic_tts_ml](https://huggingface.co/datasets/thennal/indic_tts_ml) |
| Mozilla Common Voice — Malayalam | Crowdsourced reads | varies by release | CC0 | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/ml/datasets) |
