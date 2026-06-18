# QA Agent — multimodal video checker

Two entry points share the same service layer:

- **Web page**: `/ideas/qa-video` on the button-man site → multipart upload to
  `POST /api/qa/run`.
- **CLI**: `python qa_video.py --jobId job_001` reads from
  `qa-inputs/{jobId}/`.

Both call the same pipeline (ffmpeg → Whisper → dHash similarity → GPT-4o vision
→ aggregator). The external video-generation platform is NOT touched.

## Install (local)

```bash
pip install -r requirements.txt
```

### ffmpeg (required at runtime)

- **Windows (winget)**: `winget install Gyan.FFmpeg`
- **Windows (choco)**: `choco install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

Verify: `ffmpeg -version` and `ffprobe -version`.

The Fly.dev container installs ffmpeg in its Dockerfile, so deployed runs work
without any extra setup beyond the OPENAI_API_KEY secret below.

### API key

Local:

```
cp .env.example .env
# then edit .env
OPENAI_API_KEY=sk-...
```

Fly.dev:

```bash
fly secrets set OPENAI_API_KEY=sk-... -a button-man-caify
fly deploy
```

If the key is missing, `POST /api/qa/run` returns HTTP 503 with a clear
ConfigError message instead of silently falling back to mock.

## Pipeline (service layer)

| Step | Module | Function |
|------|--------|----------|
| 1. Extract audio | `qa/ffmpeg_utils.py` | `extract_audio` |
| 2. STT | `qa/stt.py` | `transcribe_audio` (Whisper-1, verbose_json) |
| 3. Audio language | `qa/language.py` | `detect_audio_language` |
| 4. Capture frames (representative + dense) | `qa/ffmpeg_utils.py` | `capture_frames`, `capture_dense_frames` |
| 5. Frame similarity | `qa/similarity.py` | `calculate_frame_similarity` (dHash) |
| 6. Duplicate clustering | `qa/similarity.py` | `detect_duplicate_scenes` |
| 7. Visual QA | `qa/analyzer.py` | `run_visual_qa` (GPT-4o vision) |
| 8. Aggregate | `qa/aggregator.py` | `aggregate_qa_results` |

Each step is swappable: replace the function and keep the dict contract.

## Input (CLI)

```
qa-inputs/{jobId}/
├── video.mp4               (required)
├── client_info.json        (required)
├── qa_context.json         (optional — videoType/sceneIntent/expectedSubjects/forbiddenObjects)
├── script.txt              (optional)
├── scenes.json             (optional, treated as text)
└── generation_prompt.txt   (optional)
```

`client_info.json`:

```json
{
  "clientName": "강남동물의료센터",
  "industry": "동물병원",
  "services": ["피부질환", "외이질환", "심장질환"],
  "promotionPoints": ["심장질환 전문"],
  "forbiddenClaims": ["완치 보장", "100% 치료"],
  "brandTone": "전문적이고 신뢰감 있는 톤"
}
```

`qa_context.json` (optional):

```json
{
  "videoType": "병원 소개 홍보영상",
  "sceneIntent": "진료실 내부 + 수의사 진료 장면",
  "expectedSubjects": ["수의사", "진료대", "반려견"],
  "forbiddenObjects": ["자동차", "공장", "주방"]
}
```

## Run

CLI:
```bash
python qa_video.py --jobId job_001
```

Web: open `/ideas/qa-video`, fill the form, upload mp4, hit "QA 실행".

## Output (`qa_result.json` / API response)

```jsonc
{
  "jobId": "job_001",
  "status": "pass" | "retry" | "human_review" | "fail",
  "score": 0..100,
  "criticalIssues": [],
  "warnings": [],

  // text/audio flags
  "detectedForeignLanguage": false,
  "detectedForeignLanguageTTS": false,
  "detectedCompanyMixing": false,
  "detectedUnsupportedClaim": false,
  "detectedWrongIndustry": false,
  "detectedAudioScriptMismatch": false,
  "detectedVisualTextIssue": false,

  // multimodal flags
  "detectedFloatingObjects": false,
  "detectedSpatialDistortion": false,
  "detectedIrrelevantObjects": false,
  "detectedDuplicateScenes": false,

  // multimodal payload
  "sttText": "...",
  "sttLanguage": "ko",
  "sceneDiversityScore": 92.5,
  "duplicateSceneRanges": [],
  "visualAnomalyFrames": [],
  "irrelevantObjectFindings": [],
  "audioLanguageSummary": {
    "primary": "ko", "confidence": 0.95,
    "detectedSecondary": [], "foreignSegments": [], "sttMode": "whisper-1"
  },
  "visualQaSummary": [
    {"offsetSec": 0.0, "observations": [...], "spatialOk": true, "floatingObjects": [], "score": 92}
  ],
  "sceneQaSummary": [],
  "frameSummary": [
    // CLI: {path, offsetSec}
    // Web: {offsetSec, imageDataUrl}
  ],
  "retryPrompt": "",
  "humanReviewReason": "",
  "checkedAt": "2026-06-18T...Z"
}
```

## Decision rules (priority order in aggregator)

1. Non-Korean primary language or any foreign segment → `retry`.
2. 2+ medium/high `floating_furniture` anomaly frames → `retry`.
3. Irrelevant objects (industry/sceneIntent mismatch) → `retry`.
4. `sceneDiversityScore < 40` with duplicate ranges → `retry`.
5. Other-company brand on screen → `retry`.
6. Unsupported specific claim → `retry`.
7. Industry mismatch → `retry`.
8. Single floating/distortion or `detectedVisualTextIssue` → `human_review`.
9. 3+ warnings → `human_review`.
10. Otherwise → `pass`.

## Cost notes (real mode)

Per QA run on a ~30s video, ballpark:
- Whisper-1: ~$0.003 (30s of audio)
- GPT-4o vision, 6 frames "low" detail: ~$0.02
- Total: ~$0.025

Longer videos scale roughly linearly with Whisper. The dense-frame hashing is
local (no API).

## Exit codes (CLI)

| code | meaning |
|------|---------|
| 0    | pass / human_review |
| 2    | missing input file |
| 3    | unhandled error (ffmpeg missing, API failure, etc.) |
| 10   | retry |
| 11   | fail (unusable video) |
