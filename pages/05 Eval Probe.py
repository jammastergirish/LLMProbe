import gc
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import torch

from utils.load import (
    load_model_and_tokenizer,
    get_hidden_states_batched,
    is_decoder_only_model,
)

st.set_page_config(page_title="LLMProbe — Run Probe", layout="wide")


# ----------------- Helpers -----------------
def list_saved_runs(root="saved_data"):
    """
    Return a list of run folders that contain probe_weights.json and parameters.json.
    """
    candidates = sorted(
        [p for p in glob.glob(os.path.join(root, "**"), recursive=False) if os.path.isdir(p)]
    )
    runs = []
    for p in candidates:
        pw = os.path.join(p, "probe_weights.json")
        prm = os.path.join(p, "parameters.json")
        if os.path.exists(pw) and os.path.exists(prm):
            runs.append(p)
    return sorted(runs)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_probe_weights(run_folder: str):
    """
    Read probe_weights.json (metadata) and load the actual .npy weights/bias files.
    Returns: list of {'weight': np.ndarray(shape=(hidden_dim,)), 'bias': float} per layer.
    """
    meta_path = os.path.join(run_folder, "probe_weights.json")
    meta = json.load(open(meta_path, "r"))

    # Sort keys like 'layer_0', 'layer_1', ... by index
    def layer_key(k):
        m = re.search(r"layer_(\d+)", k)
        return int(m.group(1)) if m else 1_000_000

    layers = []
    for key in sorted(meta.keys(), key=layer_key):
        info = meta[key]

        # Load weights
        w_file = info["weights_file"]
        w = np.load(os.path.join(run_folder, w_file))  # shapes: (1, D) or (D,)
        if w.ndim == 2:
            if 1 in w.shape:
                w = w.reshape(-1)  # (1, D) -> (D,) or (D,1) -> (D,)
            else:
                # Expect a single-output probe; otherwise pick the first row
                w = w[0].reshape(-1)

        # Load bias if present, else 0.0
        b = 0.0
        b_file = info.get("bias_file")
        if b_file:
            b_arr = np.load(os.path.join(run_folder, b_file))
            b = float(b_arr.item()) if b_arr.size == 1 else float(np.squeeze(b_arr)[0])

        layers.append({"weight": w.astype(np.float32), "bias": b})

    return layers


def release_gpu_memory():
    """Force free GPU memory held by previous models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_device_options():
    opts = []
    if torch.cuda.is_available():
        opts.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        opts.append("mps")
    opts.append("cpu")
    return opts


# ----------------- UI -----------------
st.markdown("""
<style>
.main-title { font-size: 2.0rem; font-weight: 700; color: #FFFFFF; padding-bottom: .5rem; border-bottom: 2px solid #FFFFFF; margin-bottom: 1rem; }
.section-header { font-size: 1.2rem; font-weight: 600; color: #FFFFFF; margin-top: 1rem; }
</style>
<div class="main-title">Run a Saved Truth Probe</div>
""", unsafe_allow_html=True)

# Sidebar: pick run + device
st.sidebar.header("Configuration")

saved_runs = list_saved_runs("saved_data")
if not saved_runs:
    st.sidebar.error("No runs found. Make sure runs with probe_weights.json exist under saved_data/")
    st.stop()

run_labels = [os.path.basename(r) for r in saved_runs]
run_choice_label = st.sidebar.selectbox("📁 Saved Run", run_labels, index=len(run_labels) - 1)  # default to newest-ish
run_folder = saved_runs[run_labels.index(run_choice_label)]

params_path = os.path.join(run_folder, "parameters.json")
probes_path = os.path.join(run_folder, "probe_weights.json")
results_path = os.path.join(run_folder, "results.json")

params = load_json(params_path)
probes = load_probe_weights(run_folder)  # pass the folder, not the JSON

model_name = params.get("model_name", "unknown")
output_layer_strategy = params.get("output_activation") or params.get("embedding_strategy") or params.get(
    "output_layer")

st.sidebar.write("**Model:**", model_name)
st.sidebar.write("**Embedding:**", output_layer_strategy or "—")
st.sidebar.write("**Run Folder:**", run_choice_label)

# Device
device_name = st.sidebar.selectbox("💻 Compute", get_device_options())
device = torch.device(device_name)

# Which layer?
layer_choices = ["all"] + list(range(len(probes)))
default_layer = np.argmax(load_json(results_path)["accuracies"]) if os.path.exists(results_path) else 0


def load_model_managed(model_name: str, device: torch.device):
    """
    Load model into st.session_state.
    If a different model is already loaded, free it first.
    """
    # Check if we already have a model loaded
    if "model_name" in st.session_state:
        if st.session_state["model_name"] == model_name and st.session_state["device"] == str(device):
            # Reuse existing
            return st.session_state["tokenizer"], st.session_state["model"]

        # Different model or device → unload first
        old_model = st.session_state.get("model")
        old_tok = st.session_state.get("tokenizer")
        try:
            if old_model is not None:
                old_model.to("cpu")  # move off GPU
            del old_model, old_tok
        except Exception:
            pass
        release_gpu_memory()
        st.session_state.clear()

    # Load fresh model
    tokenizer, model = load_model_and_tokenizer(
        model_name, lambda *args, **kwargs: None, device=device
    )
    model.eval()

    # Store in session_state
    st.session_state["model_name"] = model_name
    st.session_state["device"] = str(device)
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer

    return tokenizer, model


# Load model (lazy)
with st.spinner("Loading model..."):
    tokenizer, model = load_model_managed(model_name, device)

st.divider()

# ----------------- Inference form -----------------
st.subheader("Classify a statement")
with st.form("run_probe_form", clear_on_submit=False):
    user_text = st.text_area(
        "Enter a statement:",
        placeholder="e.g., 'The Great Wall of China is visible from space with the naked eye.'",
        height=120,
    )
    submitted = st.form_submit_button("Run Probe")

# ----------------- Run inference -----------------
if submitted:
    if not user_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    ex = [{"text": user_text.strip(), "label": 0}]  # label unused for inference

    # Get ALL layers at once
    with st.spinner("Encoding and extracting hidden states for all layers..."):
        feats_all, _ = get_hidden_states_batched(
            ex, model, tokenizer, model_name,
            output_layer_strategy if output_layer_strategy else (
                "resid_post" if is_decoder_only_model(model_name) else "CLS"),
            dataset_type="INFER",
            return_layer=None,  # <-- all layers
            progress_callback=lambda *args, **kwargs: None,
            batch_size=1,
            device=device
        )  # shape: [1, num_layers, hidden_dim]

    feats_all = feats_all[0].detach().float().cpu().numpy()  # [num_layers, hidden_dim]

    rows = []
    for li in range(len(probes)):
        feats_np = feats_all[li].reshape(-1)  # (hidden_dim,)
        layer_probe = probes[li]
        w = layer_probe["weight"].astype(np.float32)
        b = float(layer_probe["bias"])

        # Align dims if needed
        if w.shape[0] != feats_np.shape[0]:
            hd = feats_np.shape[0]
            if w.shape[0] > hd:
                w = w[:hd]
            else:
                w = np.pad(w, (0, hd - w.shape[0]), constant_values=0.0)

        logit = float(np.dot(w, feats_np) + b)

        prob = float(sigmoid(logit))
        rows.append({"Layer": li, "Confidence": prob})

    chart_data = pd.DataFrame(rows)
    chart_data = chart_data.set_index("Layer")
    st.write(f"Probability of **True**")
    st.bar_chart(chart_data)
