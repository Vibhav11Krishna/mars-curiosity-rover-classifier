from PIL import Image
import pandas as pd
import streamlit as st
from sidebar import EXAMPLES_DIR, predict_top_k

try:
    from sidebar import render_page_header
except ImportError:
    def render_page_header(badge: str, title: str, subtitle: str):
        st.caption(badge)
        st.title(title)
        st.write(subtitle)

ctx = st.session_state.get("mars_ctx")
if ctx is None:
    st.error(
        "Model context unavailable. Run `notebooks/mars_classifier.ipynb` first to generate `outputs/best_mars_resnet18.pt`."
    )
    st.stop()

label_names = ctx["label_names"]

render_page_header(
    badge="MSL Curiosity · Live Inference",
    title="Mars Surface & Rover Classifier",
    subtitle="Upload a Curiosity rover image or pick a saved test sample to inspect top predictions, confidence scores, and optional label verification.",
)

with st.container(border=True):
    image_source = st.radio(
        "Image Source",
        ["Upload an image", "Use a saved test sample"],
        horizontal=True,
    )

    selected_image = None
    caption = "Curiosity Rover Image"

    if image_source == "Upload an image":
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded is not None:
            selected_image = Image.open(uploaded).copy()
            caption = f"Uploaded: {uploaded.name}"
    else:
        sample_files = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
        if sample_files:
            chosen = st.selectbox("Select saved test sample", [p.name for p in sample_files])
            selected_image = Image.open(EXAMPLES_DIR / chosen).copy()
            caption = f"Sample: {chosen}"
        else:
            st.info("No saved sample images found in `outputs/gradio_examples`. Switch to **Upload an image**.")

if selected_image is not None:
    expected_choice = st.selectbox(
        "Optional: Select actual/expected category to verify prediction",
        options=["Unknown / Unlabeled"] + [f"[{i:02d}] {name}" for i, name in enumerate(label_names)],
    )

    st.markdown("")
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        with st.container(border=True):
            st.image(selected_image, caption=caption, width="stretch")

    with col2:
        top_preds = predict_top_k(
            selected_image,
            ctx["model"],
            label_names,
            ctx["device"],
            k=ctx["top_k"],
        )
        best_label, best_prob, best_idx = top_preds[0]
        top_k_indices = [idx for _, _, idx in top_preds]

        st.metric(
            label="Top-1 Prediction",
            value=f"[{best_idx:02d}] {best_label}",
            delta=f"{best_prob:.2%} confidence",
        )

        if expected_choice != "Unknown / Unlabeled":
            expected_idx = int(expected_choice.split("]")[0].strip("["))
            expected_name = label_names[expected_idx]
            expected_prob = next((p for n, p, _ in top_preds if n == expected_name), None)

            if best_idx == expected_idx:
                st.success(f"Correct Top-1 prediction! Matched **{expected_name}** ({best_prob:.2%}).")
            elif expected_idx in top_k_indices:
                rank_pos = top_k_indices.index(expected_idx) + 1
                st.warning(
                    f"Top-1 was **{best_label}** ({best_prob:.2%}), but **{expected_name}** appeared at **Top-{rank_pos}** ({expected_prob:.2%})."
                )
            else:
                st.error(
                    f"Missed in Top-{ctx['top_k']}. Top-1 was **{best_label}** ({best_prob:.2%}) vs. actual **{expected_name}**."
                )
        else:
            if best_prob >= 0.75:
                st.info(f"High confidence on **{best_label}** ({best_prob:.2%}). Verify visually for custom/new photos.")
            else:
                st.info(f"Moderate/low top-1 confidence ({best_prob:.2%}). Inspect runner-up classes below.")

        with st.container(border=True):
            st.markdown(f"#### Top {ctx['top_k']} Predicted Classes")
            rows = []
            for rank, (cls_name, prob, cls_idx) in enumerate(top_preds, start=1):
                c_left, c_right = st.columns([3, 1])
                with c_left:
                    st.write(f"**{rank}. [{cls_idx:02d}] {cls_name}**")
                with c_right:
                    st.write(f"`{prob:.2%}`")
                st.progress(min(max(prob, 0.0), 1.0))
                rows.append({
                    "Rank": rank,
                    "Class ID": cls_idx,
                    "Category": cls_name,
                    "Confidence": round(prob, 4),
                })

            with st.expander("View scores table"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")