"""Evaluate a trained model on labeled validation/test cases.

Reports foreground Dice, IoU, precision, recall, HD95 (mm), and voxel accuracy.
Voxel accuracy includes background and can look deceptively high, so interpret it
alongside the per-structure metrics. Official ACDC testing scans often have no GT;
metrics are only computed when image/label pairs exist.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from preprocessing_and_model import (
    DEFAULT_PROJECT_DIR, DEFAULT_CHECKPOINT, PATCH_SIZE, LABEL_MAP,
    get_labeled_transforms, load_model,
)

def surface(binary):
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    eroded = ndi.binary_erosion(binary, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    return binary ^ eroded

def hd95_mm(pred, target, spacing=(1.5, 1.5, 3.0)):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")
    pred_surface = surface(pred)
    target_surface = surface(target)
    dt_target = ndi.distance_transform_edt(~target_surface, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
    return float(np.percentile(distances, 95))

def calculate_case_metrics(pred, target):
    eps = 1e-8
    metrics = {}
    voxel_correct = int((pred == target).sum())
    total_voxels = int(target.size)
    metrics["voxel_accuracy"] = voxel_correct / max(total_voxels, 1)

    for class_id in (1, 2, 3):
        p = pred == class_id
        t = target == class_id
        tp = int(np.logical_and(p, t).sum())
        fp = int(np.logical_and(p, ~t).sum())
        fn = int(np.logical_and(~p, t).sum())
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        metrics[LABEL_MAP[class_id]] = {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "hd95_mm": hd95_mm(p, t),
        }
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    split_path = Path(args.project_dir) / "splits" / f"{args.split}_files.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with split_path.open() as f:
        cases = json.load(f)
    if not cases:
        raise RuntimeError(
            f"{split_path} contains no labeled cases. If this is the official ACDC "
            "testing set without ground-truth masks, metrics cannot be computed; use inference.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, device = load_model(args.checkpoint, device)
    dataset = Dataset(data=cases, transform=get_labeled_transforms())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    per_case = []
    with torch.inference_mode():
        for idx, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            label = batch["label"][0, 0].cpu().numpy().astype(np.uint8)
            logits = sliding_window_inference(
                image, roi_size=PATCH_SIZE, sw_batch_size=1,
                predictor=model, overlap=0.5
            )
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            # Shapes should match because image and label share the same deterministic transforms.
            if pred.shape != label.shape:
                raise ValueError(
                    f"Prediction/label shape mismatch for case {idx}: "
                    f"{pred.shape} vs {label.shape}"
                )
            metrics = calculate_case_metrics(pred, label)
            case_name = os.path.basename(cases[idx - 1]["image"])
            per_case.append({"case": case_name, **metrics})
            print(f"[{idx}/{len(loader)}] {case_name} | voxel accuracy={metrics['voxel_accuracy']:.4f}")

    print(f"\nCheckpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"{'Structure':<12} {'Dice':>8} {'IoU':>8} {'Precision':>11} {'Recall':>8} {'HD95 mm':>10}")
    for name in ("RV", "MYO", "LV"):
        vals = [c[name] for c in per_case]
        print(f"{name:<12} "
              f"{np.nanmean([v['dice'] for v in vals]):8.4f} "
              f"{np.nanmean([v['iou'] for v in vals]):8.4f} "
              f"{np.nanmean([v['precision'] for v in vals]):11.4f} "
              f"{np.nanmean([v['recall'] for v in vals]):8.4f} "
              f"{np.nanmean([v['hd95_mm'] for v in vals]):10.2f}")
    print(f"\nMean voxel accuracy (includes background): "
          f"{np.mean([c['voxel_accuracy'] for c in per_case]):.4f}")

    output_path = args.output or str(Path(args.project_dir) / "outputs" / f"{args.split}_metrics.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(per_case, f, indent=2, allow_nan=True)
    print(f"Detailed per-case metrics saved to: {output_path}")

if __name__ == "__main__":
    main()
