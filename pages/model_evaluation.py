import streamlit as st

st.title("Model & Evaluation")
st.write(
    "A pretrained **ResNet-18** was fine-tuned using `AdamW` (`lr=3e-4`, `weight_decay=1e-4`), `CosineAnnealingLR`, and data augmentations (`RandomHorizontalFlip`, `ColorJitter`), saving the best checkpoint (`outputs/best_mars_resnet18.pt`) by **Validation Macro F1**."
)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Best Epoch", "10", border=True)
r2.metric("Best Val Macro F1", "0.6996", border=True)
r3.metric("Test Accuracy", "0.7372", border=True)
r4.metric("Test Weighted F1", "0.7432", border=True)

st.metric("Test Macro F1", "0.6395", border=True)

st.markdown("### Metric Summary")
st.write("- **Test Accuracy (`0.7372`):** Overall top-1 accuracy on the held-out test split.")
st.write("- **Test Weighted F1 (`0.7432`):** Support-weighted F1 across all classes.")
st.write("- **Test Macro F1 (`0.6395`):** Equal-weight average F1 across all 20 classes.")