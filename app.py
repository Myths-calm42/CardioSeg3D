"""Polished Streamlit UI for MONAI ACDC 3D cardiac MRI segmentation."""
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.colors import ListedColormap
from monai.inferers import sliding_window_inference

from preprocessing_and_model import (
    DEFAULT_CHECKPOINT, PATCH_SIZE, LABEL_MAP,
    get_image_transforms, load_model,
)

st.set_page_config(
    page_title="CardioSeg | ACDC MRI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Styling ----
st.markdown("""
<style>
    .stApp { background: #0b1220; }
    [data-testid="stHeader"] { background: rgba(11,18,32,0.88); }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 22px;
        background: linear-gradient(120deg, #172554 0%, #0f766e 100%);
        border: 1px solid rgba(255,255,255,.12); margin-bottom: 1.25rem;
    }
    .hero h1 { color: white; font-size: 2.25rem; margin: 0; letter-spacing: -.03em; }
    .hero p { color: #dbeafe; margin: .55rem 0 0 0; font-size: 1rem; }
    .eyebrow { color: #99f6e4; text-transform: uppercase; font-size: .75rem;
               font-weight: 700; letter-spacing: .16em; margin-bottom: .45rem; }
    .metric-card {
        background: #111c30; border: 1px solid #26364f; border-radius: 16px;
        padding: 1rem 1.1rem; min-height: 98px;
    }
    .metric-label { color: #a9b9d0; font-size: .82rem; margin-bottom: .35rem; }
    .metric-value { color: #f8fafc; font-size: 1.25rem; font-weight: 700; }
    .section-title { color: #f8fafc; font-size: 1.2rem; font-weight: 700; margin: .5rem 0 .75rem; }
    .small-muted { color: #a9b9d0; font-size: .85rem; }
    .stAlert { border-radius: 12px; }
    div[data-testid="stFileUploader"] { background: #111c30; padding: 1rem; border-radius: 14px; }
    div[data-testid="stDownloadButton"] button { width: 100%; }
</style>
""", unsafe_allow_html=True)

COLORS = ListedColormap(["#ff4b4b", "#31d6a0", "#36a3ff"])
LABEL_COLORS = {1: "#ff4b4b", 2: "#31d6a0", 3: "#36a3ff"}

@st.cache_resource(show_spinner=False)
def cached_model(checkpoint_path):
    return load_model(checkpoint_path)

def display_intensity(img):
    """Robustly scale a 2D slice for display only; model input is unchanged."""
    img = np.asarray(img, dtype=np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)

def plot_panel(image2d, prediction2d, mode):
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    if mode == "MRI":
        ax.imshow(display_intensity(image2d), cmap="gray", origin="lower", interpolation="nearest")
    elif mode == "Predicted labels":
        discrete_cmap = ListedColormap(["#440154", "#ff4b4b", "#31d6a0", "#36a3ff"])
        ax.imshow(prediction2d, cmap=discrete_cmap, vmin=-0.5, vmax=3.5,
                  origin="lower", interpolation="nearest")
    else:
        ax.imshow(display_intensity(image2d), cmap="gray", origin="lower", interpolation="nearest")
        overlay = np.ma.masked_where(prediction2d == 0, prediction2d - 1)
        ax.imshow(overlay, cmap=COLORS, vmin=0, vmax=2, alpha=0.58,
                  origin="lower", interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🫀 CardioSeg")
    st.caption("MONAI · 3D U-Net · ACDC")
    st.markdown("---")
    checkpoint_path = st.text_input("Model checkpoint", value=DEFAULT_CHECKPOINT)
    st.markdown("**Label guide**")
    st.markdown(
        '<span style="color:#9ca3af">■</span> 0 — Background<br>'
        '<span style="color:#ff4b4b">■</span> 1 — RV<br>'
        '<span style="color:#31d6a0">■</span> 2 — MYO<br>'
        '<span style="color:#36a3ff">■</span> 3 — LV',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Research prototype only — not for clinical diagnosis.")

# ---- Hero ----
st.markdown("""
<div class="hero">
  <div class="eyebrow">Cardiac imaging research workspace</div>
  <h1>ACDC MRI Segmentation</h1>
  <p>Explore 3D cardiac MRI volumes and inspect predicted right ventricle,
  myocardium, and left ventricle masks.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.45, 1])
with left:
    st.markdown('<div class="section-title">1 · Upload MRI volume</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a 3D NIfTI image", type=["nii", "gz"],
        help="Supported files: .nii and .nii.gz",
        label_visibility="collapsed",
    )
    st.markdown('<div class="small-muted">The volume is preprocessed using the project pipeline before inference.</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="section-title">2 · Run segmentation</div>', unsafe_allow_html=True)
    st.info("Upload a volume to enable inference. Processing time depends on volume size and available hardware.")

if uploaded is None:
    st.markdown("---")
    st.markdown("### Workflow")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**01 · Preprocess**  \nOrientation, spacing and intensity normalization")
    c2.markdown("**02 · Infer**  \n3D U-Net with sliding-window inference")
    c3.markdown("**03 · Inspect**  \nBrowse slices and download predictions")
    st.stop()

suffix = ".nii.gz" if uploaded.name.lower().endswith(".nii.gz") else ".nii"
if not uploaded.name.lower().endswith((".nii", ".nii.gz")):
    st.error("Please upload a valid .nii or .nii.gz file.")
    st.stop()

run = st.button("▶  Run 3D segmentation", type="primary", use_container_width=True)
if not run:
    st.caption(f"Selected file: **{uploaded.name}** · Click **Run 3D segmentation** to begin.")
    st.stop()

input_path = None
try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        input_path = tmp.name

    with st.status("Preparing MRI and running model inference…", expanded=True) as status:
        st.write("Loading checkpoint and preprocessing volume")
        model, checkpoint, device = cached_model(checkpoint_path)
        sample = get_image_transforms()({"image": input_path})
        image_tensor = sample["image"].unsqueeze(0).to(device)
        st.write(f"Running sliding-window inference on **{device}**")
        with torch.inference_mode():
            logits = sliding_window_inference(
                image_tensor, roi_size=PATCH_SIZE, sw_batch_size=1,
                predictor=model, overlap=0.5
            )
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        image = sample["image"][0].cpu().numpy()
        status.update(label="Segmentation complete", state="complete", expanded=False)

    if image.shape != pred.shape:
        st.error(f"Image/prediction shape mismatch: {image.shape} vs {pred.shape}")
        st.stop()

    present = np.unique(pred).tolist()
    voxel_count = int(pred.size)
    foreground_count = int(np.count_nonzero(pred))
    fg_pct = 100.0 * foreground_count / max(voxel_count, 1)
    epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"

    st.success("Inference completed successfully.")
    st.markdown('<div class="section-title">Prediction summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Checkpoint epoch", str(epoch)),
        ("Volume after preprocessing", " × ".join(map(str, image.shape))),
        ("Labels detected", f"{len(present)} / 4"),
        ("Foreground voxels", f"{fg_pct:.1f}%"),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Slice viewer</div>', unsafe_allow_html=True)
    st.caption("Use the slider to browse the preprocessed volume. These coordinates may differ from the original MRI.")
    z = st.slider("Spatial slice", 0, image.shape[-1] - 1, image.shape[-1] // 2)
    img2d = image[:, :, z].T
    pred2d = pred[:, :, z].T

    v1, v2, v3 = st.columns(3)
    for col, title, mode in [
        (v1, "MRI", "MRI"),
        (v2, "Predicted labels", "Predicted labels"),
        (v3, "Segmentation overlay", "Overlay"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            fig = plot_panel(img2d, pred2d, mode)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Explicit legend explaining class IDs and colors.
    st.markdown("#### Segmentation color legend")
    st.caption(
        "The class-map and overlay use these colors. Background (class 0) is purple "
        "in the class-map panel and transparent in the overlay."
    )
    legend_cols = st.columns(4)
    legend_items = [
        ("#440154", "Class 0", "Background"),
        ("#ff4b4b", "Class 1", "RV — Right ventricle"),
        ("#31d6a0", "Class 2", "MYO — Myocardium"),
        ("#36a3ff", "Class 3", "LV — Left ventricle"),
    ]
    for col, (color, class_id, description) in zip(legend_cols, legend_items):
        with col:
            st.markdown(
                f'<div class="metric-card" style="min-height:75px;padding:.75rem">'
                f'<div class="metric-label"><span style="color:{color};font-size:1.15rem">■</span> '
                f'<strong>{class_id}</strong></div>'
                f'<div class="small-muted">{description}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-title">Detected structures</div>', unsafe_allow_html=True)
    structure_cols = st.columns(3)
    for col, label_id in zip(structure_cols, [1, 2, 3]):
        count = int(np.count_nonzero(pred == label_id))
        percent = 100.0 * count / max(voxel_count, 1)
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">'
            f'<span style="color:{LABEL_COLORS[label_id]}">●</span> {LABEL_MAP[label_id]} '
            f'(label {label_id})</div><div class="metric-value">{percent:.2f}%</div>'
            f'<div class="small-muted">{count:,} voxels in preprocessed volume</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Export prediction</div>', unsafe_allow_html=True)
    st.caption("The download contains the preprocessed image and predicted labels. It is not restored to original MRI coordinates.")
    npz_path = Path(tempfile.gettempdir()) / "acdc_prediction_bundle.npz"
    np.savez_compressed(
        npz_path,
        image=image.astype(np.float32),
        prediction=pred,
        labels=np.array([LABEL_MAP[i] for i in range(4)]),
        target_spacing=np.array([1.5, 1.5, 3.0], dtype=np.float32),
        input_file=uploaded.name,
    )
    st.download_button(
        "⬇  Download prediction bundle (.npz)",
        data=npz_path.read_bytes(),
        file_name=Path(uploaded.name).name.replace(".nii.gz", "").replace(".nii", "") + "_prediction.npz",
        mime="application/octet-stream",
        use_container_width=True,
    )

except Exception as exc:
    st.error(f"Inference failed: {exc}")
    st.info("Verify that the checkpoint exists and matches the architecture used during training, and that the upload is a valid 3D NIfTI volume.")
finally:
    if input_path:
        try:
            os.remove(input_path)
        except OSError:
            pass
