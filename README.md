# Speech Understanding Assignment Pipeline (PA2)

Python + PyTorch implementation for code-switched lecture processing:
- Part I: STT + frame-level LID
- Part II: Hinglish -> Santhali translation + IPA mapping
- Part III: Voice cloning style synthesis (22.05 kHz output)
- Part IV: Anti-spoofing + adversarial robustness report outputs

##  Status

This repository includes a working end-to-end pipeline that runs on CPU and generates:
- `outputs/output_LRL_cloned.wav`
-  JSON artifacts for all 4 parts

## Project Layout

```text
B22CS051_PA1/
├── pipeline.py
├── requirements.txt
├── outputs/
├── src/
│   ├── stt_module.py
│   ├── lid_model.py
│   ├── translator.py
│   ├── ipa_mapper.py
│   ├── tts_module.py
│   ├── prosody_dtw.py
│   ├── anti_spoof.py
│   └── adversarial.py
├── original_segment.wav
└── student_voice_ref.wav
```

## Setup

### 1) Create and activate environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Verify required input files

Keep these files in project root (default behavior in `pipeline.py`):
- `original_segment.wav`
- `student_voice_ref.wav`

## Run

Default run:

```bash
python pipeline.py
```


## Pipeline Flow (Implemented)

### Part I - STT + LID
- Uses `STTModule` for transcription
- Uses `LanguageIdentifier` for switch-point detection
- Saves: `outputs/01_transcription_with_lid.json`

### Part II - Translation + IPA
- Uses `CodeSwitchTranslator` with custom Santhali glossary
- Uses `CodeSwitchIPAMapper` for IPA conversion
- Saves: `outputs/02_ipa_translation.json`

### Part III - TTS / Voice Cloning
- Uses `VoiceCloningPipeline` and reference voice embedding
- Synthesizes translated Santhali text to waveform
- Saves:
  - `outputs/output_LRL_cloned.wav`
  - `outputs/03_voice_cloning.json`

### Part IV - Security/Robustness
- Anti-spoof prediction using LFCC-based module
- Lightweight adversarial perturbation + SNR report
- Saves:
  - `outputs/04_spoofing_detection.json`
  - `outputs/05_adversarial_robustness.json`

## Output Files

After successful run, expected outputs:

```text
outputs/
├── 01_transcription_with_lid.json
├── 02_ipa_translation.json
├── 03_voice_cloning.json
├── 04_spoofing_detection.json
├── 05_adversarial_robustness.json
└── output_LRL_cloned.wav
```

## Notes on Output

- Translator is dictionary-based (Hinglish -> Santhali-like tokens).
- Pipeline includes a fallback to ensure synthesized text remains Santhali-leaning.
- expand `src/translator.py` dictionary for better lexical coverage.



