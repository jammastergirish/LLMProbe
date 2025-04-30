#!/bin/bash

clear

if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing with official install script..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

uv run --with streamlit streamlit run Home.py