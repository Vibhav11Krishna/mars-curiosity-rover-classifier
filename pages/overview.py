import streamlit as st

ctx = st.session_state.get("mars_ctx")
num_classes = len(ctx["label_names"]) if ctx else 20
meta = ctx["meta"] if ctx else {"epoch": 10, "best_val_macro_f1": 0.6996, "device": "cpu"}

st.title("Project Overview — Mars Curiosity Rover Image Classifier")
st.write(
    "This project (`project-mars`) is an end-to-end **Mars Curiosity Rover Image Classifier** built with "
    "**PyTorch (`ResNet-18`)**, **Hugging Face** (`RAAArity/Mars_surface_image_Curiosity_rover_labeled_data_set_version_1`), "
    "**Gradio** (in the notebook), and a multi-page **Streamlit** application."
)

st.divider()

# --- KEY SNAPSHOT METRICS ---
st.subheader("Key Project Snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Backbone Model", "ResNet-18", border=True)
m2.metric("Curiosity Classes", f"{num_classes}", border=True)
m3.metric("Train Split Size", "3,746 images", border=True)
m4.metric("Input Resolution", "224 x 224 RGB", border=True)

e1, e2, e3, e4 = st.columns(4)
e1.metric("Best Epoch", f"{meta['epoch']}", border=True)
e2.metric("Best Val Macro F1", f"{meta['best_val_macro_f1']:.4f}", border=True)
e3.metric("Test Accuracy", "0.7372", border=True)
e4.metric("Test Weighted F1", "0.7432", delta="Test Macro F1: 0.6395", border=True)

st.divider()

# --- SIMPLE EXPLANATION ---
st.subheader("What It Does (In Simple Words)")
st.write(
    "- **What goes in:** A calibrated image captured by NASA's Mars Science Laboratory (**Curiosity rover**) on Mars."
)
st.write(
    "- **What the model does:** A fine-tuned `ResNet-18` convolutional neural network analyzes the image and scores it across **20 Mars terrain and rover hardware/calibration categories** (such as `ground`, `horizon`, `inlet`, `drill`, `drill holes`, `mastcam`, `mastcam cal target`, `mahli`, `observation tray`, `portion tube`, `rems uv sensor`, `rover rear deck`, `scoop`, and `drt` classes)."
)
st.write(
    "- **What comes out:** The top predicted categories with confidence scores so you can inspect both the primary classification and alternative possibilities."
)

st.divider()

# --- WHY THIS SETUP & WHAT WAS SOLVED ---
st.subheader("Why This Setup & Engineering Highlights")
st.write(
    "- **Compact Dataset via Hugging Face:** Switched from a ~6.5 GB pixel-level terrain segmentation dataset to a ~61 MB Curiosity rover image classification dataset (`RAAArity/Mars_surface_image_Curiosity_rover_labeled_data_set_version_1`), enabling faster local experimentation and caching."
)
st.write(
    "- **Windows & Snapshot Reliability:** Used `huggingface_hub.snapshot_download` with a short cache directory (`C:\\hf_cache`) and resilient path/split parsing so dataset loading avoids Windows path-length and builder issues."
)
st.write(
    "- **Handling Class Imbalance:** Because `ground` (`2,343` training images) is far more common than rare hardware/calibration classes like `drt front` (`4`) or `apxs cal target` (`6`), model checkpointing tracks **Validation Macro F1** (`0.6996`) alongside **Accuracy** (`0.7372`), **Weighted F1** (`0.7432`), and **Test Macro F1** (`0.6395`)."
)
st.write(
    "- **Saved Weights & Dual Interfaces:** Saves the best model weights to `outputs/best_mars_resnet18.pt` and supports both standalone/Gradio testing in `notebooks/mars_classifier.ipynb` and multi-page exploration in Streamlit."
)

st.divider()

# --- END-TO-END WORKFLOW ---
st.subheader("End-to-End Workflow")
st.write("1. **Download & Inspect:** Fetch the Curiosity dataset snapshot, parse `train`/`val`/`test` splits, and inspect class counts and sample images.")
st.write("2. **Preprocess & Augment:** Resize images to `224x224`, normalize with ImageNet mean/std, and apply `RandomHorizontalFlip` and `ColorJitter` during training.")
st.write("3. **Fine-Tune & Checkpoint:** Train pretrained `ResNet-18` with `AdamW` (`lr=3e-4`, `weight_decay=1e-4`) and `CosineAnnealingLR`, saving `outputs/best_mars_resnet18.pt` when Validation Macro F1 improves.")
st.write("4. **Evaluate & Serve:** Evaluate the saved checkpoint on the held-out test split and serve predictions interactively in Streamlit and Gradio.")

st.divider()

# --- PROJECT STRUCTURE & COMMANDS ---
col_struct, col_cmd = st.columns(2)

with col_struct:
    st.subheader("Project Structure")
    st.code(
        """project-2-mars/
├── app.py
├── sidebar.py
├── pages/
│   ├── classifier.py
│   ├── overview.py
│   ├── dataset_info.py
│   └── model_evaluation.py
├── notebooks/
│   └── mars_classifier.ipynb
└── outputs/
    ├── best_mars_resnet18.pt
    └── gradio_examples/""",
        language="text",
    )

with col_cmd:
    st.subheader("How to Run")
    st.code(
        """# From inside project-2-mars/
python -m streamlit run app.py""",
        language="powershell",
    )
    st.write(
        "Use the left sidebar to switch between **Live Classifier**, **Project Overview**, "
        "**Dataset & Imbalance**, and **Model & Evaluation**, or adjust the **Top-K predictions** slider."
    )