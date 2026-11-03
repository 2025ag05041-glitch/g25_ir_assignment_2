"""Entry point: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="G25 Text Mining & Document Intelligence", layout="wide")

pages = [
    st.Page("views/dashboard.py", title="Dashboard", default=True),
    st.Page("views/dataset_explorer.py", title="Dataset & Upload"),
    st.Page("views/preprocessing.py", title="Preprocessing & Features"),
    st.Page("views/text_mining.py", title="Text Mining (NB & K-Means)"),
    st.Page("views/recommendations.py", title="Recommendations"),
    st.Page("views/performance.py", title="Performance Analysis"),
]
navigation = st.navigation(pages)

from src.app_state import sidebar_status  # noqa: E402

sidebar_status()
st.sidebar.caption("IR Assignment 2 - Group G25")

navigation.run()
