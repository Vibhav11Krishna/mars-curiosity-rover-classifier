import pandas as pd
import streamlit as st
from sidebar import TRAIN_CLASS_COUNTS

st.title("Dataset & Class Imbalance")
st.write(
    "Dataset: `RAAArity/Mars_surface_image_Curiosity_rover_labeled_data_set_version_1` across `train`, `validation`, and `test` splits (**3,746** training images across **20** classes)."
)

df = pd.DataFrame(TRAIN_CLASS_COUNTS, columns=["Class ID", "Category", "Train Samples"])
df["Share (%)"] = (df["Train Samples"] / df["Train Samples"].sum() * 100).round(2)

m1, m2, m3 = st.columns(3)
m1.metric("Total Train Samples", f"{df['Train Samples'].sum():,}", border=True)
m2.metric("Most Common Class", "ground (2,343)", border=True)
m3.metric("Smallest Class", "drt front (4)", border=True)

st.markdown("### Training Samples per Class")
st.bar_chart(df.set_index("Category")["Train Samples"], width="stretch")

st.markdown("### 20-Class Distribution Table")
st.dataframe(df, hide_index=True, width="stretch")

st.info(
    "`ground` makes up over 62% of the training set, while classes like `drt front` (4), `apxs cal target` (6), `drt side` (8), and `portion tube opening` (8) have very few samples. **Macro F1** helps keep rare classes visible in evaluation."
)