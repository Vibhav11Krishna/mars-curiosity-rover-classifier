from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = PROJECT_ROOT / "outputs" / "best_mars_resnet18.pt"
EXAMPLES_DIR = PROJECT_ROOT / "outputs" / "gradio_examples"

TRAIN_CLASS_COUNTS = [
    (0, "apxs", 30),
    (1, "apxs cal target", 6),
    (2, "chemcam cal target", 15),
    (3, "chemin inlet open", 47),
    (4, "drill", 19),
    (5, "drill holes", 36),
    (6, "drt front", 4),
    (7, "drt side", 8),
    (8, "ground", 2343),
    (9, "horizon", 224),
    (10, "inlet", 165),
    (11, "mahli", 14),
    (12, "mahli cal target", 32),
    (13, "mastcam", 36),
    (14, "mastcam cal target", 101),
    (15, "observation tray", 85),
    (16, "portion box", 18),
    (17, "portion tube", 73),
    (18, "portion tube opening", 8),
    (19, "rems uv sensor", 12),
]

DEFAULT_LABEL_NAMES = [name for _, name, _ in TRAIN_CLASS_COUNTS]


def apply_mars_theme():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2.5rem;
                max-width: 1180px;
            }
            .mars-hero {
                background: linear-gradient(135deg, rgba(194, 65, 12, 0.22), rgba(124, 45, 18, 0.10));
                border: 1px solid rgba(251, 146, 60, 0.28);
                border-radius: 14px;
                padding: 1.15rem 1.35rem;
                margin-bottom: 1.1rem;
            }
            .mars-hero-title {
                font-size: 1.55rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin: 0 0 0.25rem 0;
            }
            .mars-hero-sub {
                font-size: 0.94rem;
                opacity: 0.85;
                margin: 0;
            }
            .mars-badge {
                display: inline-block;
                font-size: 0.74rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                padding: 0.22rem 0.6rem;
                border-radius: 999px;
                background: rgba(234, 88, 12, 0.22);
                border: 1px solid rgba(251, 146, 60, 0.4);
                color: #fdba74;
                margin-bottom: 0.55rem;
            }
            .mars-card {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 12px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.75rem;
            }
            .mars-card-label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                opacity: 0.72;
                margin-bottom: 0.2rem;
            }
            .mars-card-value {
                font-size: 1.05rem;
                font-weight: 600;
            }
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 12px;
                padding: 0.8rem 1rem;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(148, 163, 184, 0.16);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(badge: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="mars-hero">
            <div class="mars-badge">{badge}</div>
            <div class="mars-hero-title">{title}</div>
            <p class="mars-hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_checkpoint(ckpt_path_str: str, device: torch.device) -> dict:
    try:
        return torch.load(ckpt_path_str, map_location=device, weights_only=True)
    except Exception:
        return torch.load(ckpt_path_str, map_location=device, weights_only=False)


@st.cache_resource
def load_mars_model(ckpt_path_str: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(ckpt_path_str, device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        label_names = ckpt.get("label_names") or DEFAULT_LABEL_NAMES
        num_classes = int(ckpt.get("num_classes", len(label_names)))
        epoch = ckpt.get("epoch", 10)
        best_val_macro_f1 = float(ckpt.get("best_val_macro_f1", 0.6996))
    else:
        state_dict = ckpt
        label_names = DEFAULT_LABEL_NAMES
        num_classes = len(label_names)
        epoch = 10
        best_val_macro_f1 = 0.6996

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {
        "epoch": epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "device": str(device),
    }
    return model, list(label_names), device, meta


@torch.no_grad()
def predict_top_k(image: Image.Image, model, label_names, device, k: int = 5):
    rgb = image.convert("RGB")
    tensor = eval_transform(rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_k = min(max(int(k), 1), len(label_names))
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(label_names[i], float(probs[i]), int(i)) for i in top_indices]


def render_sidebar():
    apply_mars_theme()

    if not DEFAULT_CKPT.exists():
        st.sidebar.error(f"Checkpoint not found at `{DEFAULT_CKPT}`.")
        return None

    try:
        model, label_names, device, meta = load_mars_model(str(DEFAULT_CKPT))
    except Exception as exc:
        st.sidebar.error(f"Failed to load checkpoint: {exc}")
        return None

    with st.sidebar:
        st.markdown("### MSL Curiosity Classifier")
        st.caption("PyTorch ResNet-18 · 20 Classes")
        st.divider()

        st.markdown(
            f"""
            <div class="mars-card">
                <div class="mars-card-label">Runtime Device</div>
                <div class="mars-card-value">{meta['device'].upper()}</div>
            </div>
            <div class="mars-card">
                <div class="mars-card-label">Best Checkpoint</div>
                <div class="mars-card-value">Epoch {meta['epoch']} · F1 {meta['best_val_macro_f1']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Test Acc", "73.72%")
        with m_col2:
            st.metric("Macro F1", "0.6395")

        st.metric("Weighted F1", "0.7432")
        st.divider()

        top_k = st.slider(
            "Top-K predictions",
            min_value=1,
            max_value=min(10, len(label_names)),
            value=5,
            key="shared_top_k",
        )

    return {
        "model": model,
        "label_names": label_names,
        "device": device,
        "meta": meta,
        "top_k": top_k,
    }