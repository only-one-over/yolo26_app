"""Summarize locally recorded cold/hot startup timing events.

Run the application once after clearing ``logs/startup_metrics.jsonl`` for a cold sample,
then run it again for a hot sample. This tool produces JSON and CSV summaries without
opening a GUI, so it works for source and onedir package logs alike.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("startup_benchmark"))
    parser.add_argument("--mode", choices=("cold", "hot"), default="hot")
    args = parser.parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in args.metrics.read_text("utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        grouped[event.get("runtime", "unknown")].append(event)
    runs: list[dict] = []
    current: list[dict] = []
    for runtime_events in grouped.values():
        for event in runtime_events:
            if event.get("stage") == "process_entry" and current:
                runs.append(current)
                current = []
            current.append(event)
    if current:
        runs.append(current)
    summaries = []
    for events in runs:
        by_stage = {event.get("stage"): event for event in events}
        summaries.append({
            "mode": args.mode,
            "runtime": events[0].get("runtime", "unknown"),
            "window_visible_ms": by_stage.get("main_window_shown", {}).get("elapsed_ms", ""),
            "interactive_ms": by_stage.get("main_window_interactive", {}).get("elapsed_ms", ""),
            "project_restored_ms": by_stage.get("app_state_restored", {}).get("elapsed_ms", ""),
            "index_ready_ms": by_stage.get("media_index_ready", {}).get("elapsed_ms", ""),
            "thumbnail_ready_ms": by_stage.get("first_thumbnail_ready", {}).get("elapsed_ms", ""),
            "media_count": by_stage.get("media_index_ready", {}).get("media_count", ""),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), "utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]) if summaries else ["mode"])
        writer.writeheader()
        writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
