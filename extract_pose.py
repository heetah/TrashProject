#!/usr/bin/env python3
"""Build MMACTION2/ST-GCN pose annotations from dataset-pose videos.

The output ``garbage_final.pkl`` follows MMACTION2 ``PoseDataset`` format:

    {
        "annotations": [
            {
                "frame_dir": "normal_case1",
                "label": 0,
                "img_shape": (h, w),
                "total_frames": T,
                "keypoint": np.ndarray[M, T, 17, 2],
                "keypoint_score": np.ndarray[M, T, 17],
            },
            ...
        ],
        "split": {"train": [...], "val": [...]}
    }

Labels are intentionally aligned with scripts-old-test/action.py:
0 = normal, 1 = littering.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


LABEL_TO_ID = {
    "normal": 0,
    "littering": 1,
}


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def dump_pickle(obj: object, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mmengine

        mmengine.dump(obj, str(out_path))
    except Exception:
        with out_path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> object:
    try:
        import mmengine

        return mmengine.load(str(path))
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def collect_videos(dataset_dir: Path, limit: int | None = None) -> list[dict]:
    samples = []
    for label_name, label_id in LABEL_TO_ID.items():
        class_dir = dataset_dir / label_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for video_path in sorted(class_dir.glob("*.mp4"), key=natural_key):
            samples.append(
                {
                    "path": video_path,
                    "frame_dir": video_path.stem,
                    "label_name": label_name,
                    "label": label_id,
                }
            )

    seen = {}
    for sample in samples:
        frame_dir = sample["frame_dir"]
        if frame_dir in seen:
            raise ValueError(
                f"Duplicate frame_dir '{frame_dir}': {seen[frame_dir]} and {sample['path']}"
            )
        seen[frame_dir] = sample["path"]

    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError(f"No mp4 files found under {dataset_dir}")
    return samples


def filter_readable_videos(
    samples: list[dict], dataset_dir: Path, skip_bad_videos: bool
) -> list[dict]:
    readable = []
    failures = []
    for sample in samples:
        try:
            probe_video(sample["path"])
            readable.append(sample)
        except Exception as exc:
            failures.append((sample["path"].as_posix(), repr(exc)))

    if failures:
        failure_path = dataset_dir / "bad_videos.json"
        failure_path.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not skip_bad_videos:
            bad_list = "\n".join(path for path, _ in failures[:10])
            raise RuntimeError(
                f"Found {len(failures)} unreadable videos. "
                f"Re-run with --skip-bad-videos to exclude them.\n{bad_list}"
            )
        print(f"Excluded {len(failures)} unreadable videos. See {failure_path}")

    if not readable:
        raise RuntimeError("No readable videos remain after filtering")
    return readable


def build_stratified_split(
    samples: list[dict], train_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []

    for label_id in sorted(LABEL_TO_ID.values()):
        group = [s["frame_dir"] for s in samples if s["label"] == label_id]
        rng.shuffle(group)
        split_idx = int(len(group) * train_ratio)
        train_ids.extend(group[:split_idx])
        val_ids.extend(group[split_idx:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def write_text_outputs(
    samples: list[dict],
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    dataset_dir: Path,
    out_path: Path,
    raw_out_path: Path,
    args: argparse.Namespace,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    list_specs = {
        "all": samples,
        "train": [s for s in samples if s["frame_dir"] in train_id_set],
        "val": [s for s in samples if s["frame_dir"] in val_id_set],
    }
    for split_name, split_samples in list_specs.items():
        list_path = dataset_dir / f"{split_name}_videos.txt"
        lines = [
            f"{sample['path'].as_posix()} {sample['label']}"
            for sample in split_samples
        ]
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    label_lines = [
        f"{label_id} {label_name}"
        for label_name, label_id in sorted(LABEL_TO_ID.items(), key=lambda x: x[1])
    ]
    (dataset_dir / "label_map.txt").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8"
    )

    counts = {}
    for sample in samples:
        counts.setdefault(sample["label_name"], 0)
        counts[sample["label_name"]] += 1

    metadata = {
        "dataset_dir": dataset_dir.as_posix(),
        "raw_annotations": raw_out_path.as_posix(),
        "final_annotations": out_path.as_posix(),
        "label_to_id": LABEL_TO_ID,
        "num_videos": len(samples),
        "class_counts": counts,
        "split_counts": {
            "train": len(train_id_set),
            "val": len(val_id_set),
        },
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "pose_model": args.pose_model,
        "max_person": args.max_person,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (dataset_dir / "stgcn_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def probe_video(video_path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video shape for {video_path}: {height}x{width}")
    return height, width, max(total_frames, 0)


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def iter_progress(items: list[dict], desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except Exception:
        print(desc)
        return items


def extract_one_video(
    model,
    sample: dict,
    cache_path: Path,
    args: argparse.Namespace,
    device: str,
) -> dict:
    if cache_path.exists() and not args.overwrite_cache:
        return load_pickle(cache_path)

    video_path = sample["path"]
    height, width, probed_total_frames = probe_video(video_path)
    predict_kwargs = {
        "source": str(video_path),
        "stream": True,
        "conf": args.conf,
        "device": device,
        "verbose": False,
    }
    if args.imgsz > 0:
        predict_kwargs["imgsz"] = args.imgsz
    if args.half and device != "cpu":
        predict_kwargs["half"] = True

    keypoints_per_frame: list[np.ndarray] = []
    scores_per_frame: list[np.ndarray] = []

    for result in model.predict(**predict_kwargs):
        frame_keypoints = np.zeros((args.max_person, 17, 2), dtype=np.float16)
        frame_scores = np.zeros((args.max_person, 17), dtype=np.float16)

        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None and keypoints.xy is not None:
            xy = keypoints.xy.detach().cpu().numpy()
            if xy.size:
                if getattr(keypoints, "conf", None) is not None:
                    conf = keypoints.conf.detach().cpu().numpy()
                else:
                    conf = np.ones(xy.shape[:2], dtype=np.float32)

                person_order = np.arange(xy.shape[0])
                boxes = getattr(result, "boxes", None)
                if boxes is not None and boxes.conf is not None:
                    box_conf = boxes.conf.detach().cpu().numpy()
                    if box_conf.shape[0] == xy.shape[0]:
                        person_order = np.argsort(-box_conf)
                else:
                    person_order = np.argsort(-conf.mean(axis=1))

                for out_idx, person_idx in enumerate(person_order[: args.max_person]):
                    frame_keypoints[out_idx] = xy[person_idx].astype(np.float16)
                    frame_scores[out_idx] = conf[person_idx].astype(np.float16)

        keypoints_per_frame.append(frame_keypoints)
        scores_per_frame.append(frame_scores)

    if not keypoints_per_frame:
        raise RuntimeError(f"No frames decoded by YOLO for {video_path}")

    total_frames = len(keypoints_per_frame)
    if probed_total_frames and abs(probed_total_frames - total_frames) > 1:
        print(
            f"Warning: frame count mismatch for {video_path}: "
            f"ffprobe/cv2={probed_total_frames}, yolo={total_frames}",
            file=sys.stderr,
        )

    keypoint = np.stack(keypoints_per_frame, axis=1)
    keypoint_score = np.stack(scores_per_frame, axis=1)
    valid_pose_frames = int((keypoint_score.max(axis=(0, 2)) > 0).sum())

    annotation = {
        "frame_dir": sample["frame_dir"],
        "label": int(sample["label"]),
        "img_shape": (height, width),
        "original_shape": (height, width),
        "total_frames": int(total_frames),
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,
        "valid_pose_frames": valid_pose_frames,
    }
    dump_pickle(annotation, cache_path)
    return annotation


def build_annotations(samples: list[dict], args: argparse.Namespace) -> list[dict]:
    from ultralytics import YOLO

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"Loading pose model: {args.pose_model}")
    print(f"Pose extraction device: {device}")
    model = YOLO(args.pose_model)

    annotations = []
    failures = []
    for sample in iter_progress(samples, "extract pose"):
        cache_name = f"{sample['label_name']}__{sample['frame_dir']}.pkl"
        cache_path = cache_dir / cache_name
        try:
            annotations.append(extract_one_video(model, sample, cache_path, args, device))
        except Exception as exc:
            failures.append((sample["path"].as_posix(), repr(exc)))
            if not args.skip_bad_videos:
                raise

    if failures:
        failure_path = Path(args.dataset_dir) / "pose_extraction_failures.json"
        failure_path.write_text(
            json.dumps(failures, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Skipped {len(failures)} bad videos. See {failure_path}")

    return annotations


def validate_final_pkl(path: Path) -> None:
    data = load_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path} should contain a dict, got {type(data)}")
    annotations = data.get("annotations")
    split = data.get("split")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(f"{path} has no annotations")
    if not isinstance(split, dict) or "train" not in split or "val" not in split:
        raise ValueError(f"{path} split must contain train and val")

    required = {"frame_dir", "label", "total_frames", "img_shape", "keypoint", "keypoint_score"}
    labels = {}
    for anno in annotations:
        missing = required - set(anno)
        if missing:
            raise ValueError(f"{anno.get('frame_dir')} is missing {sorted(missing)}")
        keypoint = anno["keypoint"]
        keypoint_score = anno["keypoint_score"]
        if keypoint.ndim != 4 or keypoint.shape[-2:] != (17, 2):
            raise ValueError(f"{anno['frame_dir']} bad keypoint shape: {keypoint.shape}")
        if keypoint_score.shape != keypoint.shape[:-1]:
            raise ValueError(
                f"{anno['frame_dir']} score shape {keypoint_score.shape} "
                f"does not match keypoint shape {keypoint.shape}"
            )
        labels.setdefault(int(anno["label"]), 0)
        labels[int(anno["label"])] += 1

    print(
        f"Validated {path}: {len(annotations)} annotations, "
        f"train={len(split['train'])}, val={len(split['val'])}, labels={labels}"
    )


def summarize_split_frames(data: dict, split_name: str) -> dict:
    split_ids = set(data["split"][split_name])
    stats: dict[int, dict[str, int]] = {}
    for anno in data["annotations"]:
        if anno["frame_dir"] not in split_ids:
            continue
        label = int(anno["label"])
        stats.setdefault(label, {"samples": 0, "frames": 0})
        stats[label]["samples"] += 1
        stats[label]["frames"] += int(anno["total_frames"])
    return {str(label): stats[label] for label in sorted(stats)}


def best_subset_by_total_frames(annotations: list[dict], target_frames: int) -> list[dict]:
    if target_frames <= 0:
        return []

    # Choose the closest subset that does not exceed target_frames. This keeps
    # the oversampled class from becoming larger than the reference class.
    dp: dict[int, tuple[int, ...]] = {0: ()}
    for idx, anno in enumerate(annotations):
        frames = int(anno["total_frames"])
        if frames <= 0:
            continue
        for current, chosen in list(dp.items()):
            candidate = current + frames
            if candidate <= target_frames and candidate not in dp:
                dp[candidate] = chosen + (idx,)

    best_total = max(dp)
    return [annotations[idx] for idx in dp[best_total]]


def clone_balanced_annotation(
    anno: dict,
    duplicate_index: int,
    existing_ids: set[str],
) -> dict:
    clone = {}
    for key, value in anno.items():
        if isinstance(value, np.ndarray):
            clone[key] = value.copy()
        else:
            clone[key] = copy.deepcopy(value)

    source_frame_dir = anno.get("source_frame_dir", anno["frame_dir"])
    while True:
        frame_dir = f"{source_frame_dir}__balanced_dup_{duplicate_index:04d}"
        duplicate_index += 1
        if frame_dir not in existing_ids:
            break

    existing_ids.add(frame_dir)
    clone["frame_dir"] = frame_dir
    clone["source_frame_dir"] = source_frame_dir
    clone["balanced_duplicate"] = True
    return clone


def build_frame_balanced_pose_data(data: dict, split_name: str = "train") -> tuple[dict, dict]:
    if any(anno.get("balanced_duplicate") for anno in data["annotations"]):
        raise ValueError(
            "Input annotations already contain balanced duplicates. "
            "Use the original unbalanced pkl as --out when running --balance-only."
        )

    split_ids = set(data["split"][split_name])
    by_label: dict[int, list[dict]] = {}
    for anno in data["annotations"]:
        if anno["frame_dir"] in split_ids:
            by_label.setdefault(int(anno["label"]), []).append(anno)

    if len(by_label) < 2:
        raise ValueError(f"Need at least two labels in split '{split_name}' to balance")

    before_stats = summarize_split_frames(data, split_name)
    frame_totals = {
        label: sum(int(anno["total_frames"]) for anno in annos)
        for label, annos in by_label.items()
    }
    target_frames = max(frame_totals.values())
    existing_ids = {anno["frame_dir"] for anno in data["annotations"]}
    duplicate_counts: dict[str, int] = {}
    duplicate_annos: list[dict] = []
    duplicate_split_ids: list[str] = []
    duplicate_report: dict[str, dict[str, int]] = {}

    for label in sorted(by_label):
        label_frames = frame_totals[label]
        if label_frames <= 0:
            raise ValueError(f"Label {label} has no frames in split '{split_name}'")
        deficit = target_frames - label_frames
        if deficit <= 0:
            duplicate_report[str(label)] = {
                "source_frames": label_frames,
                "target_frames": target_frames,
                "added_samples": 0,
                "added_frames": 0,
                "full_repeats": 0,
                "subset_samples": 0,
            }
            continue

        full_repeats, remaining_frames = divmod(deficit, label_frames)
        selected: list[dict] = []
        for _ in range(full_repeats):
            selected.extend(by_label[label])
        subset = best_subset_by_total_frames(by_label[label], remaining_frames)
        selected.extend(subset)

        added_frames = 0
        for anno in selected:
            source = anno.get("source_frame_dir", anno["frame_dir"])
            duplicate_counts[source] = duplicate_counts.get(source, 0) + 1
            duplicate = clone_balanced_annotation(
                anno, duplicate_counts[source], existing_ids
            )
            duplicate_annos.append(duplicate)
            duplicate_split_ids.append(duplicate["frame_dir"])
            added_frames += int(duplicate["total_frames"])

        duplicate_report[str(label)] = {
            "source_frames": label_frames,
            "target_frames": target_frames,
            "added_samples": len(selected),
            "added_frames": added_frames,
            "full_repeats": full_repeats,
            "subset_samples": len(subset),
        }

    balanced_data = {
        "annotations": data["annotations"] + duplicate_annos,
        "split": copy.deepcopy(data["split"]),
    }
    balanced_data["split"][split_name] = (
        list(data["split"][split_name]) + duplicate_split_ids
    )

    after_stats = summarize_split_frames(balanced_data, split_name)
    metadata = {
        "balance_method": "invisible_annotation_duplication",
        "split": split_name,
        "target_frames": int(target_frames),
        "before": before_stats,
        "after": after_stats,
        "duplicates": duplicate_report,
        "added_annotations": len(duplicate_annos),
    }
    return balanced_data, metadata


def write_frame_balanced_pose_data(
    in_path: Path,
    out_path: Path,
    metadata_path: Path,
    split_name: str = "train",
) -> None:
    data = load_pickle(in_path)
    if not isinstance(data, dict):
        raise TypeError(f"{in_path} should contain a dict, got {type(data)}")

    balanced_data, metadata = build_frame_balanced_pose_data(data, split_name)
    metadata["input_annotations"] = in_path.as_posix()
    metadata["balanced_annotations"] = out_path.as_posix()

    dump_pickle(balanced_data, out_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote frame-balanced annotations: {out_path}")
    print(f"Wrote balance metadata: {metadata_path}")
    for label, stats in metadata["after"].items():
        print(
            f"label={label} split={split_name} "
            f"samples={stats['samples']} frames={stats['frames']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MMACTION2 PoseDataset annotations for trash ST-GCN training."
    )
    parser.add_argument("--dataset-dir", default="dataset-pose")
    parser.add_argument("--pose-model", default="modules_weight/yolo26x-pose.pt")
    parser.add_argument("--raw-out", default="dataset-pose/garbage_raw.pkl")
    parser.add_argument("--out", default="dataset-pose/garbage_final.pkl")
    parser.add_argument("--cache-dir", default="dataset-pose/pose_cache")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-person", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-bad-videos", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--balance-only", action="store_true")
    parser.add_argument("--balance-train-frames", action="store_true")
    parser.add_argument("--balance-out", default="dataset-pose/garbage_final_balanced.pkl")
    parser.add_argument(
        "--balance-metadata-out",
        default="dataset-pose/stgcn_balanced_metadata.json",
    )
    parser.add_argument("--balance-split", default="train")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)
    raw_out_path = Path(args.raw_out)
    balance_out_path = Path(args.balance_out)
    balance_metadata_path = Path(args.balance_metadata_out)

    if args.validate_only:
        validate_final_pkl(out_path)
        return
    if args.balance_only:
        write_frame_balanced_pose_data(
            out_path, balance_out_path, balance_metadata_path, args.balance_split
        )
        validate_final_pkl(balance_out_path)
        return
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    samples = collect_videos(dataset_dir, limit=args.limit)
    samples = filter_readable_videos(samples, dataset_dir, args.skip_bad_videos)
    train_ids, val_ids = build_stratified_split(samples, args.train_ratio, args.seed)
    write_text_outputs(samples, train_ids, val_ids, dataset_dir, out_path, raw_out_path, args)
    print(
        f"Prepared lists: total={len(samples)}, train={len(train_ids)}, val={len(val_ids)}"
    )

    if args.list_only:
        return

    extraction_samples = samples[args.shard_index :: args.num_shards]
    if args.num_shards > 1:
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{len(extraction_samples)} videos"
        )

    annotations = build_annotations(extraction_samples, args)
    if args.cache_only:
        print(f"Cached {len(annotations)} annotations; final pkl not written in cache-only mode.")
        return

    if args.num_shards != 1:
        raise RuntimeError(
            "Shard extraction should use --cache-only. "
            "Run this script once more without sharding to merge caches."
        )

    annotation_by_id = {anno["frame_dir"]: anno for anno in annotations}
    available_train = [frame_dir for frame_dir in train_ids if frame_dir in annotation_by_id]
    available_val = [frame_dir for frame_dir in val_ids if frame_dir in annotation_by_id]
    available_samples = [
        sample for sample in samples if sample["frame_dir"] in annotation_by_id
    ]
    ordered_annotations = [
        annotation_by_id[sample["frame_dir"]] for sample in available_samples
    ]

    if len(available_samples) != len(samples):
        write_text_outputs(
            available_samples,
            available_train,
            available_val,
            dataset_dir,
            out_path,
            raw_out_path,
            args,
        )

    dump_pickle(ordered_annotations, raw_out_path)
    final_data = {
        "annotations": ordered_annotations,
        "split": {
            "train": available_train,
            "val": available_val,
        },
    }
    dump_pickle(final_data, out_path)
    validate_final_pkl(out_path)
    if args.balance_train_frames:
        write_frame_balanced_pose_data(
            out_path, balance_out_path, balance_metadata_path, args.balance_split
        )
        validate_final_pkl(balance_out_path)


if __name__ == "__main__":
    main()
