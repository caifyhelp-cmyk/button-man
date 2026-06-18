# QA Agent — local video checker

Stand-alone CLI that audits a locally downloaded marketing video against
`client_info.json` and produces `qa_result.json`. Does **not** call the external
video-generation platform.

## Install

```bash
pip install -r requirements.txt
```

### ffmpeg (required)

The tool shells out to `ffmpeg` and `ffprobe` to extract audio and sample frames.

- **Windows (winget)**: `winget install Gyan.FFmpeg`
- **Windows (choco)**: `choco install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

Verify: `ffmpeg -version` and `ffprobe -version`.

### API key

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=sk-...
```

(Whisper handles STT; GPT-4o vision handles the QA judgment.)

## Input layout

```
qa-inputs/
└── {jobId}/
    ├── video.mp4               (required)
    ├── client_info.json        (required)
    ├── script.txt              (optional — intended narration)
    ├── scenes.json             (optional — scene metadata)
    └── generation_context.json (optional — anything else)
```

`client_info.json` shape:

```json
{
  "clientName": "강남동물의료센터",
  "industry": "동물병원",
  "services": ["피부질환", "외이질환", "심장질환", "슬개골탈구", "애견미용"],
  "promotionPoints": ["심장질환 전문", "피부질환 전문", "동천동 동물병원"],
  "forbiddenClaims": ["완치 보장", "100% 치료", "최고", "유일"],
  "brandTone": "전문적이고 신뢰감 있는 톤"
}
```

## Run

```bash
python qa_video.py --jobId job_001
```

Optional:

```bash
python qa_video.py --jobId job_001 --inputs-dir /path/to/jobs --quiet
```

## Output

- `qa-inputs/{jobId}/qa_result.json` — verdict
- `qa-inputs/{jobId}/artifacts/audio.mp3` — extracted audio
- `qa-inputs/{jobId}/artifacts/frame_*.jpg` — sampled frames (0s, 2s, 5s, 10s, 15s, end-2s)

`qa_result.json` schema:

```jsonc
{
  "jobId": "job_001",
  "status": "pass | retry | human_review | fail",
  "score": 0-100,
  "criticalIssues": [],
  "warnings": [],
  "detectedForeignLanguage": false,
  "detectedCompanyMixing": false,
  "detectedUnsupportedClaim": false,
  "detectedWrongIndustry": false,
  "detectedVisualTextIssue": false,
  "detectedAudioScriptMismatch": false,
  "sttText": "...",
  "sttLanguage": "ko",
  "frameSummary": [{"path": "artifacts/frame_0000000ms.jpg", "offsetSec": 0.0}],
  "retryPrompt": "...",
  "humanReviewReason": "",
  "checkedAt": "2026-06-18T..."
}
```

## Exit codes

| code | meaning |
|------|---------|
| 0    | pass / human_review (handle in queue, but tool succeeded) |
| 2    | missing input file |
| 3    | unhandled error (ffmpeg missing, API failure, etc.) |
| 10   | retry |
| 11   | fail (unusable video) |

## Decision rules (summary)

| signal | verdict |
|--------|---------|
| Other company name in STT or on-screen | retry |
| Unsupported specifics (numbers, certs, prices, location) | retry |
| Non-Korean TTS detected | retry |
| Industry mismatch in scenes | retry |
| Medical / legal / financial / income exaggeration, or forbiddenClaims hit | human_review |
| Broken subtitles / cropped on-screen text | human_review |
| STT vs script meaning shifted in risky way | human_review |
| Minor wording nits only | pass + warnings |
| Broken file / wrong client entirely | fail |
