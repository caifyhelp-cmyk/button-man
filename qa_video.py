#!/usr/bin/env python
"""QA Agent CLI — checks locally downloaded marketing videos against client_info."""
import argparse
import json
import sys
from pathlib import Path

from qa.runner import run_qa


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run QA on a local video file (qa-inputs/{jobId}/video.mp4)."
    )
    parser.add_argument("--jobId", required=True, dest="job_id",
                        help="Folder name under qa-inputs/ (e.g. job_001)")
    parser.add_argument("--inputs-dir", default="qa-inputs",
                        help="Root inputs directory (default: qa-inputs)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print result summary line")
    args = parser.parse_args()

    try:
        result = run_qa(args.job_id, args.inputs_dir)
    except FileNotFoundError as exc:
        print(f"[QA] ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[QA] FAILED: {exc}", file=sys.stderr)
        return 3

    summary = (
        f"[QA] {result['jobId']}: status={result['status']} "
        f"score={result['score']} "
        f"critical={len(result.get('criticalIssues', []))} "
        f"warnings={len(result.get('warnings', []))}"
    )
    print(summary)
    if not args.quiet:
        out_path = Path(args.inputs_dir) / args.job_id / "qa_result.json"
        print(f"[QA] result -> {out_path}")
        if result["status"] == "retry" and result.get("retryPrompt"):
            print("[QA] retryPrompt preview:")
            print(result["retryPrompt"][:400] + ("..." if len(result["retryPrompt"]) > 400 else ""))

    exit_map = {"pass": 0, "human_review": 0, "retry": 10, "fail": 11}
    return exit_map.get(result["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
