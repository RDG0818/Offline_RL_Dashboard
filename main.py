import streamlit as st
from utils.log_utils import logger
from backend import init_db

st.set_page_config(
    page_title="Offline RL Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📉 Offline RL Dashboard")
st.markdown("Welcome to the dashboard for analyzing offline reinforcement learning algorithms.")

st.info("Use the sidebar to navigate between pages. This is the homepage.")

init_db()
