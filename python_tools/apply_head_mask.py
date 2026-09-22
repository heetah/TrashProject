"""Post-process STGCN pkl: zero out head node (0-4) keypoint_score channels.

COCO 17-point head nodes (0-indexed):
  0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear

Zeroing the score channel suppresses these joints as input features while
preserving their x,y coords (so PreNormalize2D geometry is unaffected).
"""
import argparse
import pickle
import numpy as np
from pathlib import Path

HEAD_NODES = [0, 1, 2, 3, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input pkl path")
    parser.add_argument("output", help="output pkl path")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    anns = data["annotations"]
    print(f"Annotations: {len(anns)}")

    for ann in anns:
        ks = ann["keypoint_score"]  # (num_person, T, V)
        ks[:, :, HEAD_NODES] = 0.0
        ann["keypoint_score"] = ks

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data, f)

    # Verify
    sample = anns[0]["keypoint_score"]
    assert sample[:, :, HEAD_NODES].max() == 0.0, "head mask not applied!"
    body_mean = anns[0]["keypoint_score"][:, :, 5:].mean()
    print(f"Head nodes 0-4 score (should be 0): {sample[:, :, HEAD_NODES].max():.4f}")
    print(f"Body nodes 5-16 score mean: {body_mean:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
