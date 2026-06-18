# qa-inputs/

Drop each QA job into its own folder here:

```
qa-inputs/
└── job_001/
    ├── video.mp4           (required)
    ├── client_info.json    (required)
    ├── script.txt          (optional)
    ├── scenes.json         (optional)
    └── generation_context.json (optional)
```

Run:

```bash
python qa_video.py --jobId job_001
```

Results land at `qa-inputs/job_001/qa_result.json`.

This folder is gitignored — videos and results stay local.
