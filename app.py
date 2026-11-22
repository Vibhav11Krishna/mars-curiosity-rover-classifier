import streamlit as st
from sidebar import render_sidebar

st.set_page_config(
    page_title="Mars Curiosity Rover Classifier",
    page_icon=":material/rocket_launch:",
    layout="wide",
)

ctx = render_sidebar()
st.session_state["mars_ctx"] = ctx

pg = st.navigation(
    {
        "Application": [
            st.Page("pages/classifier.py", title="Live Classifier", icon=":material/image_search:", default=True),
        ],
        "Project Information": [
            st.Page("pages/overview.py", title="Project Overview", icon=":material/info:"),
            st.Page("pages/dataset_info.py", title="Dataset & Imbalance", icon=":material/dataset:"),
            st.Page("pages/model_evaluation.py", title="Model & Evaluation", icon=":material/analytics:"),
        ],
    },
    position="sidebar",
    expanded=True,
)
pg.run()