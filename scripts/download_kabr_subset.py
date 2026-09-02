from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "imageomics/KABR"
REPO_TYPE = "dataset"
ANNOTATION_ROOT = "KABR/annotation"
IMAGE_ROOT = "KABR/dataset/image"

CLASSES = {
    "Auto-Groom": 0,
    "Browse": 1,
    "Graze": 2,
    "Head Up": 3,
    "Occluded": 4,
    "Run": 5,
    "Trot": 6,
    "Walk": 7,
}


def get_csv(name: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{ANNOTATION_ROOT}/{name}",
        repo_type=REPO_TYPE,
        local_dir=cache,
    ))


def read_kabr_csv(path: Path) -> pd.DataFrame:
    # KABR annotation files are whitespace-delimited despite the .csv suffix.
    return pd.read_csv(path, sep=r"\s+", engine="python")


def load_annotations(cache: Path):
    train = read_kabr_csv(get_csv("train.csv", cache))
    val = read_kabr_csv(get_csv("val.csv", cache))

    required = {"original_vido_id", "video_id", "frame_id", "path", "labels"}
    for df in (train, val):
        if "original_video_id" in df.columns and "original_vido_id" not in df.columns:
            df.rename(columns={"original_video_id": "original_vido_id"}, inplace=True)
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Unexpected KABR CSV schema. Missing {sorted(missing)}; "
                f"columns={list(df.columns)}"
            )
    return train, val


def select_sequences(df, per_class, sequence_length, seed):
    rng = random.Random(seed)
    id_to_name = {v: k for k, v in CLASSES.items()}
    df = df.copy()
    df["label_id"] = pd.to_numeric(df["labels"], errors="coerce")
    df = df[df["label_id"].isin(id_to_name)].copy()

    candidates = []
    for video_id, group in df.groupby("video_id", sort=False):
        group = group.sort_values("frame_id")
        counts = group["label_id"].astype(int).value_counts()
        if counts.empty:
            continue
        dominant_id = int(counts.index[0])
        purity = float(counts.iloc[0] / len(group))
        if len(group) < sequence_length or purity < 0.75:
            continue
        candidates.append({
            "video_id": int(video_id),
            "label_id": dominant_id,
            "label": id_to_name[dominant_id],
            "purity": round(purity, 4),
            "frames": group,
        })

    by_label = {name: [] for name in CLASSES}
    for item in candidates:
        by_label[item["label"]].append(item)

    selected = []
    for label in CLASSES:
        pool = by_label[label]
        rng.shuffle(pool)
        for item in pool[:per_class]:
            frames = item["frames"]
            indices = [
                round(i * (len(frames) - 1) / (sequence_length - 1))
                for i in range(sequence_length)
            ]
            sampled = frames.iloc[indices]
            selected.append({
                "sequence_id": str(item["video_id"]),
                "video_id": item["video_id"],
                "label": item["label"],
                "label_id": item["label_id"],
                "purity": item["purity"],
                "frames": sampled["path"].tolist(),
            })
    return selected


def resolve_image_filename(dataset_path: str) -> str:
    # The annotation path is already relative to the image directory in the
    # KABR repository. Avoid duplicating IMAGE_ROOT when it is already present.
    p = str(dataset_path).replace("\\", "/").lstrip("/")
    prefix = IMAGE_ROOT.rstrip("/") + "/"
    if p.startswith(prefix):
        return p
    # KABR paths are normally e.g. ZP0627.5/1.jpg.
    return f"{IMAGE_ROOT}/{p}"


def download_frames(items, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    total = sum(len(x["frames"]) for x in items)
    done = 0

    for item in items:
        seq_dir = output_dir / "frames" / item["sequence_id"]
        seq_dir.mkdir(parents=True, exist_ok=True)
        local_frames = []

        for i, dataset_path in enumerate(item["frames"]):
            target = seq_dir / f"{i:03d}.jpg"
            if not target.exists():
                filename = resolve_image_filename(dataset_path)
                try:
                    source = hf_hub_download(
                        repo_id=REPO_ID,
                        filename=filename,
                        repo_type=REPO_TYPE,
                        local_dir=output_dir / "_hf_cache",
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not download KABR image.\n"
                        f"CSV path: {dataset_path}\n"
                        f"Attempted repository path: {filename}\n"
                        f"Original error: {exc}"
                    ) from exc
                target.write_bytes(Path(source).read_bytes())
            local_frames.append(str(target.resolve()))
            done += 1
            print(f"  [{done}/{total}] {item['label']}")

        result.append({**item, "frames": local_frames})
    return result


def write_manifest(items, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "KABR",
        "source": REPO_ID,
        "classes": CLASSES,
        "sequence_length": len(items[0]["frames"]) if items else 0,
        "items": items,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "classes.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    summary = {
        "sequences": len(items),
        "frames": sum(len(x["frames"]) for x in items),
        "class_counts": dict(Counter(x["label"] for x in items)),
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description="Create a tiny KABR behavior pilot from official ML-ready annotations.")
    p.add_argument("--per-class", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--output", default="data/raw/kabr")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output = Path(args.output)

    print("[1/4] Downloading official KABR annotation CSVs...")
    train, val = load_annotations(output / "_annotations")
    print(f"  train rows: {len(train):,}")
    print(f"  val rows:   {len(val):,}")

    print("[2/4] Selecting balanced temporal sequences...")
    train_items = select_sequences(train, args.per_class, args.sequence_length, args.seed)
    val_items = select_sequences(val, max(1, args.per_class // 2), args.sequence_length, args.seed + 1)

    train_counts = Counter(x["label"] for x in train_items)
    print("  train:", dict(train_counts))
    print("  val:", dict(Counter(x["label"] for x in val_items)))

    missing = [c for c in CLASSES if train_counts[c] < args.per_class]
    if missing:
        raise RuntimeError("No suitable KABR sequence found for: " + ", ".join(missing))

    print("[3/4] Downloading only selected KABR image frames...")
    train_local = download_frames(train_items, output / "train")
    val_local = download_frames(val_items, output / "val")

    print("[4/4] Writing manifests...")
    train_summary = write_manifest(train_local, output / "train")
    val_summary = write_manifest(val_local, output / "val")

    combined = {
        "dataset": "KABR",
        "classes": CLASSES,
        "train": str((output / "train" / "manifest.json").resolve()),
        "val": str((output / "val" / "manifest.json").resolve()),
    }
    (output / "dataset_manifest.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print("\nKABR_DOWNLOAD_OK")
    print(json.dumps({"train": train_summary, "val": val_summary}, indent=2))


if __name__ == "__main__":
    main()
