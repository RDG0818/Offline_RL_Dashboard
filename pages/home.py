import streamlit as st
from utils.plot_utils import example_plot



st.title("📊 Algorithm Overview")
st.markdown("Compare training metrics across offline RL algorithms.")

st.plotly_chart(example_plot(), use_container_width=True)
