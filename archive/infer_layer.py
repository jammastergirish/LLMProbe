# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "datasets",
#   "matplotlib",
#   "scikit-learn",
#   "protobuf",
#   "tiktoken",
#   "blobfile",
#  "accelerate",
# "transformer-lens",
# "streamlit",
#  "pandas",
# ]
# ///

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    Llama4ForConditionalGeneration,
)
from transformer_lens import HookedTransformer
import pandas as pd
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'

# uv run --with streamlit --with torch  --with transformers --with datasets --with matplotlib --with scikit-learn --with protobug --with tiktoken --with blobfile --with accelerate --with transformer-lens streamlit run  infer_layer.py


def logit_lens_eval_llama3(model, tokenizer, prompt, target_token):
    import pandas as pd

    tokens = tokenizer(prompt, return_tensors="pt")[
        "input_ids"].to(model.cfg.device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    layers = []
    probs = []
    preds = []

    for layer in range(model.cfg.n_layers):
        resid = cache["resid_post", layer]
        x = model.ln_final(resid)
        logits = model.unembed(x)
        probs_softmax = torch.softmax(logits[0, -1], dim=0)
        pred_id = probs_softmax.argmax().item()
        pred = tokenizer.decode(pred_id)
        confidence = probs_softmax[target_id].item()

        layers.append(layer)
        probs.append(confidence)
        preds.append(pred)

    # Table of predictions
    df = pd.DataFrame({
        "Layer": layers,
        "Top Prediction": preds,
        f"Prob('{target_token.strip()}')": probs
    })

    st.write("### 🔍 Output")

    # st.dataframe(df)

    chart_data = pd.DataFrame({
        "Layer": layers,
        "Confidence": probs
    })

    chart_data = chart_data.set_index("Layer")
    st.write(f"Probability of {target_token}")
    st.bar_chart(chart_data)


def logit_lens_eval_llama4(model, tokenizer, prompt, target_token):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layers = []
    probs = []
    preds = []

    for i, h in enumerate(outputs.hidden_states):
        last_token_state = h[0, -1]
        logits = model.lm_head(last_token_state)
        probs_softmax = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs_softmax).item()
        pred_token = tokenizer.decode(pred_id)


        if probs_softmax.dim() == 2:
            confidence = probs_softmax[0, target_id].item()
        else:
            confidence = probs_softmax[target_id].item()

        layers.append(i)
        probs.append(confidence)
        preds.append(pred_token)

    df = pd.DataFrame({
        "Layer": layers,
        "Top Prediction": preds,
        f"Prob('{target_token.strip()}')": probs
    })

    st.write("### 🔍 Output")
    st.dataframe(df)
    st.bar_chart(df.set_index("Layer")[f"Prob('{target_token.strip()}')"])


def logit_lens_eval_gpt2_or_deepseek(model, tokenizer, prompt, target_token):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layers = []
    probs = []
    preds = []

    for i, h in enumerate(outputs.hidden_states):
        last_token_state = h[0, -1]
        logits = model.lm_head(last_token_state)
        probs_softmax = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs_softmax).item()
        pred_token = tokenizer.decode(pred_id)
        
        if probs_softmax.dim() == 2:
            confidence = probs_softmax[0, target_id].item()
        else:
            confidence = probs_softmax[target_id].item()

        layers.append(i)
        probs.append(confidence)
        preds.append(pred_token)

    df = pd.DataFrame({
        "Layer": layers,
        "Top Prediction": preds,
        f"Prob('{target_token.strip()}')": probs
    })

    st.write("### 🔍 Output")
    st.dataframe(df)
    st.bar_chart(df.set_index("Layer")[f"Prob('{target_token.strip()}')"])


def logit_lens_eval_bert(model, tokenizer, prompt, target_token):
    # Auto-insert [MASK] if not found
    if "[MASK]" not in prompt:
        prompt = prompt.strip()
        if not prompt.endswith((".", "?", "!", ",")):
            prompt += "."
        prompt += " [MASK]"

    inputs = tokenizer(prompt, return_tensors="pt")
    mask_index = (inputs["input_ids"] ==
                  tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_index) == 0:
        st.error("❌ Could not find [MASK] token even after appending.")
        return

    target_id = tokenizer.convert_tokens_to_ids(target_token)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

    transform = model.cls

    layers = []
    probs = []
    preds = []

    for i, h in enumerate(hidden_states):
        masked_hidden = h[0, mask_index]
        transformed = transform.predictions.transform(masked_hidden)
        logits = transform.predictions.decoder(transformed)
        probs_softmax = torch.softmax(logits, dim=-1)

        pred_id = torch.argmax(probs_softmax, dim=-1).squeeze().item()
        pred_token = tokenizer.decode(pred_id)
        confidence = probs_softmax[0, target_id].item()

        layers.append(i)
        probs.append(confidence)
        preds.append(pred_token)

    df = pd.DataFrame({
        "Layer": layers,
        "Top Prediction": preds,
        f"Prob('{target_token.strip()}')": probs
    })

    st.write("### 🔍 Output")
    st.write(f"Used prompt: `{prompt}`")
    st.dataframe(df)
    st.bar_chart(df.set_index("Layer")[f"Prob('{target_token.strip()}')"])


# ------------------------ Streamlit UI ------------------------

st.title("🔍 Inference at All Layers")

prompt = st.text_input("Prompt", "The capital of France is")
target = st.text_input("Target token", " Paris")

model_choice = st.selectbox("Choose a model", [
    "meta-llama/Llama-3.2-1B-Instruct",     # TransformerLens
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",  # HF causal
    "deepseek-ai/DeepSeek-V3-Base",  # HF causal
    "gpt2",                                  # HF causal
    "bert-base-uncased",                    # HF masked
])

if st.button("Run Analysis"):
    if "llama-3" in model_choice.lower():
        model = HookedTransformer.from_pretrained(
            model_choice,
            device="cuda" if torch.cuda.is_available() else "mps"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        logit_lens_eval_llama3(model, tokenizer, prompt, target)

    elif any(x in model_choice.lower() for x in ["llama-4", "gpt2", "deepseek", "v3"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_choice,
            device_map="auto",  # https://github.com/pytorch/pytorch/issues/141287. Switch to MPS or auto when possible
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        logit_lens_eval_gpt2_or_deepseek(model, tokenizer, prompt, target)

    elif "bert" in model_choice.lower():
        model = AutoModelForMaskedLM.from_pretrained(
            model_choice, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        logit_lens_eval_bert(model, tokenizer, prompt, target)

    else:
        st.error("Model not supported.")
