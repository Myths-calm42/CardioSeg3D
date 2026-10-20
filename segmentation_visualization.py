"""Visualize a prediction NPZ produced by inference.py, or evaluate a labeled case."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COLORS = ListedColormap(["red", "lime", "deepskyblue"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="NPZ created by inference.py")
    parser.add_argument("--slice", type=int, default=None, help="Slice index; default is middle")
    parser.add_argument("--save", default=None, help="Optional output PNG path")
    args = parser.parse_args()

    bundle = np.load(args.npz, allow_pickle=True)
    image = bundle["image"]
    pred = bundle["prediction"]
    if image.shape != pred.shape:
        raise ValueError(f"Image/prediction shape mismatch: {image.shape} vs {pred.shape}")

    z = args.slice if args.slice is not None else image.shape[-1] // 2
    if not 0 <= z < image.shape[-1]:
        raise ValueError(f"Slice must be in [0, {image.shape[-1] - 1}]")

    img2d = image[:, :, z].T
    pred2d = pred[:, :, z].T
    pred_overlay = np.ma.masked_where(pred2d == 0, pred2d - 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img2d, cmap="gray", origin="lower")
    axes[0].set_title(f"Preprocessed MRI — slice {z}")
    axes[1].imshow(pred2d, cmap="viridis", vmin=0, vmax=3, origin="lower")
    axes[1].set_title("Predicted labels (0=BG, 1=RV, 2=MYO, 3=LV)")
    axes[2].imshow(img2d, cmap="gray", origin="lower")
    axes[2].imshow(pred_overlay, cmap=COLORS, vmin=0, vmax=2, alpha=0.55, origin="lower")
    axes[2].set_title("MRI + predicted RV/MYO/LV")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=160, bbox_inches="tight")
        print("Saved visualization:", args.save)
    plt.show()

if __name__ == "__main__":
    main()
