"""
download_audioset.py — Filter AudioSet CSVs for cheating-relevant labels
and download audio clips using yt-dlp.

Usage:
    python3 cheating_detection/data/download_audioset.py

Relevant AudioSet label IDs for exam cheating detection:
    /m/09x0r   Speech
    /m/01j3sz  Whispering
    /m/07plct  Breathing
    /m/0dl9sf8 Coughing
    /m/01hsr_  Sneezing
    /m/04brg2  Keyboard (typing)
    /m/07pxg6x Writing
    /m/09hlz4  Rustling
    /m/0dl83   Silence
    /m/02dgv   Door
"""

import os
import csv
import subprocess
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BALANCED_CSV = Path("/mnt/c/Users/kenne/Downloads/balanced_train_segments.csv")
EVAL_CSV     = Path("/mnt/c/Users/kenne/Downloads/eval_segments.csv")
OUTPUT_DIR   = Path(__file__).resolve().parent.parent.parent / "datasets" / "audio" / "audioset"

# ── Relevant label IDs → our exam label ──────────────────────────────────────
LABEL_MAP = {
    "/m/09x0r":   "suspicious",   # Speech
    "/m/01j3sz":  "suspicious",   # Whispering
    "/m/07plct":  "suspicious",   # Breathing
    "/m/0dl9sf8": "suspicious",   # Coughing
    "/m/01hsr_":  "suspicious",   # Sneezing
    "/m/04brg2":  "normal",       # Keyboard typing
    "/m/07pxg6x": "normal",       # Writing
    "/m/0dl83":   "normal",       # Silence
    "/m/09hlz4":  "alert",        # Rustling
    "/m/02dgv":   "alert",        # Door
}

MAX_PER_LABEL = 50   # download max 50 clips per label (adjust as needed)


def parse_audioset_csv(csv_path: Path) -> list[dict]:
    """Parse AudioSet CSV — skip comment lines starting with #."""
    rows = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",", 3)
            if len(parts) < 4:
                continue
            ytid       = parts[0].strip()
            start_sec  = float(parts[1].strip())
            end_sec    = float(parts[2].strip())
            labels     = parts[3].strip().strip('"').split(",")
            labels     = [l.strip() for l in labels]
            rows.append({
                "ytid": ytid,
                "start": start_sec,
                "end": end_sec,
                "labels": labels,
            })
    return rows


def filter_relevant(rows: list[dict]) -> list[dict]:
    """Keep only rows that contain at least one relevant label."""
    filtered = []
    for row in rows:
        for lbl in row["labels"]:
            if lbl in LABEL_MAP:
                row["exam_label"] = LABEL_MAP[lbl]
                row["matched_label"] = lbl
                filtered.append(row)
                break
    return filtered


def download_clip(ytid: str, start: float, end: float, out_path: Path) -> bool:
    """Download a 10-second clip from YouTube using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={ytid}"
    duration = end - start

    cmd = [
        "yt-dlp",
        "-x",                          # extract audio only
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"-ss {start} -t {duration}",
        "--output", str(out_path),
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True)
        return result.returncode == 0 and out_path.exists()
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in ["normal", "suspicious", "alert"]:
        (OUTPUT_DIR / label).mkdir(exist_ok=True)

    print("Parsing AudioSet CSVs...")
    rows = []
    for csv_path in [BALANCED_CSV, EVAL_CSV]:
        if csv_path.exists():
            r = parse_audioset_csv(csv_path)
            rows.extend(r)
            print(f"  Loaded {len(r)} rows from {csv_path.name}")
        else:
            print(f"  WARNING: {csv_path} not found — skipping")

    print(f"Total rows: {len(rows)}")

    relevant = filter_relevant(rows)
    print(f"Relevant rows: {len(relevant)}")

    # Count per label
    from collections import Counter
    label_counts = Counter(r["exam_label"] for r in relevant)
    print(f"  normal: {label_counts['normal']}, "
          f"suspicious: {label_counts['suspicious']}, "
          f"alert: {label_counts['alert']}")

    # Download with per-label cap
    downloaded = Counter()
    skipped    = 0
    failed     = 0

    print(f"\nDownloading (max {MAX_PER_LABEL} per label)...")
    for i, row in enumerate(relevant):
        exam_label = row["exam_label"]
        if downloaded[exam_label] >= MAX_PER_LABEL:
            continue

        out_path = OUTPUT_DIR / exam_label / f"{row['ytid']}_{int(row['start'])}.wav"
        if out_path.exists():
            downloaded[exam_label] += 1
            skipped += 1
            continue

        success = download_clip(row["ytid"], row["start"], row["end"], out_path)
        if success:
            downloaded[exam_label] += 1
            print(f"  [{sum(downloaded.values())}] ✓ {exam_label}: {row['ytid']}")
        else:
            failed += 1

        time.sleep(0.5)   # be polite to YouTube

        # Stop when all labels are saturated
        if all(downloaded[l] >= MAX_PER_LABEL for l in ["normal", "suspicious", "alert"]):
            break

    print(f"\nDownload complete:")
    print(f"  normal:     {downloaded['normal']} clips")
    print(f"  suspicious: {downloaded['suspicious']} clips")
    print(f"  alert:      {downloaded['alert']} clips")
    print(f"  failed:     {failed}")
    print(f"  Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
