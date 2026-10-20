"""Shared model and deterministic preprocessing for the ACDC 3D U-Net project."""
import os
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, ScaleIntensityRangePercentilesd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm

LABEL_MAP = {0: "Background", 1: "RV", 2: "MYO", 3: "LV"}
NUM_CLASSES = 4
PATCH_SIZE = (128, 128, 16)
TARGET_SPACING = (1.5, 1.5, 3.0)
ORIENTATION = "RAS"

DEFAULT_PROJECT_DIR = os.environ.get(
    "ACDC_PROJECT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)
DEFAULT_CHECKPOINT = os.environ.get(
    "ACDC_CHECKPOINT",
    os.path.join(DEFAULT_PROJECT_DIR, "checkpoints", "best_model.pt")
)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(device=None):
    """Must match the architecture used in ACDC_Full_Pipeline_and_Training.ipynb."""
    if device is None:
        device = get_device()
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    ).to(device)
    return model

def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    device = device or get_device()
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first, then set ACDC_CHECKPOINT to best_model.pt."
        )
    model = build_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint, device

def get_image_transforms():
    """Same deterministic image preprocessing used during training/validation."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes=ORIENTATION),
        Spacingd(keys=["image"], pixdim=TARGET_SPACING, mode="bilinear"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image"], source_key="image", margin=15, allow_smaller=True
        ),
        SpatialPadd(keys=["image"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image"]),
    ])

def get_labeled_transforms():
    """Same deterministic preprocessing used for validation, for image/GT pairs."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=ORIENTATION),
        Spacingd(
            keys=["image", "label"], pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"], source_key="image",
            margin=15, allow_smaller=True
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image", "label"]),
    ])
