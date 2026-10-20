"""Run 3D U-Net inference on a single NIfTI MRI and save a prediction bundle.

Example:
  python inference.py --image /path/to/patient001_frame01.nii.gz
  python inference.py --image scan.nii.gz --checkpoint checkpoints/best_model.pt
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from preprocessing_and_model import (
    DEFAULT_CHECKPOINT, PATCH_SIZE, LABEL_MAP,
    get_image_transforms, load_model,
)

def predict_nifti(image_path, checkpoint_path=DEFAULT_CHECKPOINT, output_dir="outputs"):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    model, checkpoint, device = load_model(checkpoint_path)
    transform = get_image_transforms()
    sample = transform({"image": str(image_path)})
    image = sample["image"].unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
        prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    processed_image = sample["image"][0].cpu().numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.name.replace('.nii.gz', '').replace('.nii', '')}_prediction.npz"

    # NPZ is saved in the preprocessed RAS/target-spacing/cropped coordinate space.
    # It is NOT automatically restored to the original NIfTI orientation/spacing.
    np.savez_compressed(
        output_path,
        image=processed_image.astype(np.float32),
        prediction=prediction,
        labels=np.array([LABEL_MAP[i] for i in range(len(LABEL_MAP))]),
        target_spacing=np.array([1.5, 1.5, 3.0], dtype=np.float32),
        input_file=str(image_path),
    )
    return output_path, prediction, processed_image, checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input 3D NIfTI (.nii or .nii.gz)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    out, pred, image, ckpt = predict_nifti(args.image, args.checkpoint, args.output_dir)
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Preprocessed image shape: {image.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Predicted labels present: {np.unique(pred).tolist()}")
    print(f"Saved prediction bundle: {out}")

if __name__ == "__main__":
    main()
