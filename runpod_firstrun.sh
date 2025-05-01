#!/bin/bash

clear

pip install uv

pip install -U "huggingface_hub[cli]"

huggingface-cli login

uv run --with streamlit streamlit run Home.py --server.port=8501 --server.address=0.0.0.0 --server.enableWebsocketCompression=false