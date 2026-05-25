"""
Extracts training metrics and duration from offline Whisper runs.

This module parses Hugging Face trainer state JSON files and
TensorBoard event files to compile a comprehensive training report
and export the step-by-step metrics to a CSV format.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from constants import (
    STATE_FILE_NAME,
    TFEVENTS_PATTERN,
    CSV_HEADERS,
    KEY_LOG_HISTORY,
    KEY_STEPS_PER_SEC,
    KEY_LOSS,
    KEY_EPOCH,
    KEY_STEP,
    UNKNOWN_TIME
)


def find_latest_state(base_dir: Path) -> Path | None:
    """
    Locates the most recent trainer_state.json within a directory.

    Checks the root directory first, and if unavailable, searches
    checkpoint subdirectories, returning the most recently modified.

    Parameters
    ----------
    base_dir : Path
        The output directory of the Whisper training run.

    Returns
    -------
    Path | None
        The path to the latest state file, or None if not found.

    Examples
    --------
    >>> run_dir = Path("/mnt/hf_cache/whisper-finetuned")
    >>> state_path = find_latest_state(base_dir=run_dir)
    >>> print(state_path.name)
    trainer_state.json
    """
    state_file: Path = base_dir / STATE_FILE_NAME
    if state_file.exists():
        return state_file

    # Locate all checkpoint states using a generator expression
    checkpoints: list[Path] = list(
        base_dir.glob(pattern=f"checkpoint-*/{STATE_FILE_NAME}")
    )

    if not checkpoints:
        return None

    # Retrieve the state file with the latest modified time
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def get_time_bounds(event_file: Path) -> tuple[float, float]:
    """
    Extracts the exact start and end timestamps from a run.

    Hugging Face encodes the start timestamp in the TensorBoard
    event filename, while the end time is the file modification time.

    Parameters
    ----------
    event_file : Path
        The path to the TensorBoard event file.

    Returns
    -------
    tuple[float, float]
        A tuple containing the start and end timestamps in seconds.

    Examples
    --------
    >>> event_path = Path("events.out.tfevents.1712080000.host")
    >>> start, end = get_time_bounds(event_file=event_path)
    """
    parts: list[str] = event_file.name.split(sep=".")
    try:
        # The Unix timestamp is typically the 4th segment
        start_ts: float = float(parts[3])
    except (IndexError, ValueError):
        # Fallback to creation time if the standard format changes
        start_ts = event_file.stat().st_ctime

    end_ts: float = event_file.stat().st_mtime
    return start_ts, end_ts


def print_time_report(start_ts: float, end_ts: float) -> None:
    """
    Formats and prints the total training duration to stdout.

    Parameters
    ----------
    start_ts : float
        The Unix timestamp representing the start of training.
    end_ts : float
        The Unix timestamp representing the end of training.

    Examples
    --------
    >>> print_time_report(start_ts=1712080000.0, end_ts=1712083600.0)
    """
    elapsed: float = end_ts - start_ts
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    start_dt: str = datetime.fromtimestamp(
        timestamp=start_ts
    ).strftime(format="%Y-%m-%d %H:%M:%S")

    end_dt: str = datetime.fromtimestamp(
        timestamp=end_ts
    ).strftime(format="%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 40)
    print("⏱️  TRAINING TIME REPORT")
    print("=" * 40)
    print(f"Start Time:      {start_dt}")
    print(f"End Time:        {end_dt}")
    print("-" * 40)
    duration_str: str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    print(f"Total Duration:  {duration_str}")
    print("=" * 40 + "\n")


def parse_and_export_metrics(state_path: Path, output_csv: Path) -> None:
    """
    Reads JSON logs and exports training metrics to a CSV file.

    Parameters
    ----------
    state_path : Path
        The path to the trainer_state.json file.
    output_csv : Path
        The destination path for the exported metrics CSV.

    Examples
    --------
    >>> state = Path("trainer_state.json")
    >>> dest = Path("metrics.csv")
    >>> parse_and_export_metrics(state_path=state, output_csv=dest)
    """
    with open(file=state_path, mode="r", encoding="utf-8") as f:
        data: dict = json.load(fp=f)

    log_history: list[dict] = data.get(KEY_LOG_HISTORY, [])

    # Isolate the training speed metric to estimate step timestamps
    steps_per_sec: float | None = None
    for entry in log_history:
        if KEY_STEPS_PER_SEC in entry:
            steps_per_sec = float(entry[KEY_STEPS_PER_SEC])

    rows: list[dict[str, float | int | str]] = []
    for entry in log_history:
        # Isolate training loss logs, ignoring evaluation logs
        if KEY_LOSS in entry and KEY_EPOCH in entry:
            epoch: float = round(number=entry[KEY_EPOCH], ndigits=4)
            loss: float = round(number=entry[KEY_LOSS], ndigits=4)
            step: int = entry.get(KEY_STEP, 0)

            elapsed: float | str = UNKNOWN_TIME
            if steps_per_sec is not None:
                elapsed = round(number=(step / steps_per_sec), ndigits=2)

            rows.append({
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "estimated_elapsed_time_sec": elapsed
            })

    if not rows:
        print("⚠️ No training loss data found in the state logs.")
        return

    # Export parsed data strictly adhering to CSV formatting
    with open(
        file=output_csv,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rowdicts=rows)

    print(f"✅ Successfully exported {len(rows)} logs to {output_csv}")


def main(output_dir: str, output_csv: str) -> None:
    """
    Orchestrates the metric extraction and reporting workflow.

    Parameters
    ----------
    output_dir : str
        The root directory of the model's Hugging Face outputs.
    output_csv : str
        The desired file path for the CSV output.

    Examples
    --------
    Run module locally using uv package manager:
    >>> uv run python extract_metrics.py --dir /mnt/out --csv out.csv
    """
    base_path: Path = Path(output_dir)
    csv_path: Path = Path(output_csv)

    # 1. Resolve duration from TensorBoard events
    events: list[Path] = list(base_path.glob(pattern=TFEVENTS_PATTERN))
    if events:
        latest_event: Path = max(events, key=lambda p: p.stat().st_mtime)
        start_ts, end_ts = get_time_bounds(event_file=latest_event)
        print_time_report(start_ts=start_ts, end_ts=end_ts)
    else:
        print(f"❌ TensorBoard logs missing from {base_path}/runs")

    # 2. Extract metrics from state logs
    state_file: Path | None = find_latest_state(base_dir=base_path)
    if state_file is None:
        print(f"❌ Could not find {STATE_FILE_NAME} in {base_path}")
        return

    parse_and_export_metrics(state_path=state_file, output_csv=csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract offline training stats from Hugging Face."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the model's output directory."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Desired destination path for the output CSV."
    )
    args: argparse.Namespace = parser.parse_args()

    main(output_dir=args.dir, output_csv=args.csv)
