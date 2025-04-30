# main.py
import streamlit as st
import torch
import platform

# Configure the page - this should be the first Streamlit command
st.set_page_config(
    page_title="LLM Analysis Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("🧠 LLM Probing Suite")

st.markdown("""
### Welcome to the LLM Analysis Suite

This application provides various tools for analyzing large language models.

Select an analysis from the sidebar to get started:

1. **Truth Detection Probing** - Analyze how LLMs encode truth and falsehood
2. **Other Analysis** - Add your own analyses here

""")

# Show information about navigation
st.info("""
    **Navigation**: Use the sidebar to switch between different analyses.
    Each page will have its own configuration options and results.
""")

st.write("### Device Information")
st.write(f"- Platform: {platform.platform()}")
st.write(f"- Python Version: {platform.python_version()}")
st.write(f"- PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    st.write(
        f"- CUDA Available: Yes (Device: {torch.cuda.get_device_name(0)})")
else:
    st.write("- CUDA Available: No")

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    st.write("- MPS Available: Yes (Apple Silicon)")
else:
    st.write("- MPS Available: No")
