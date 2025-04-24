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
# "transformer-lens"
# ]
# ///
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer


def logit_lens_eval(model: HookedTransformer, tokens: torch.Tensor, layers: list[int], target_token: str = "8"):
    """
    Apply logit lens to examine model's prediction at each layer by projecting
    the residual stream directly into the vocabulary space.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

        print(f"🔍 Target token: {repr(target_token)}\n")
        target_id = model.tokenizer.encode(
            target_token, add_special_tokens=False)[0]

        for layer in layers:
            # Get residual stream at that layer
            resid = cache["resid_post", layer]  # shape: [1, seq_len, d_model]

            # Apply final layer norm and unembedding (logit lens)
            x = model.ln_final(resid)
            logits = model.unembed(x)  # shape: [1, seq_len, vocab_size]

            # Focus on the last token position
            probs = torch.softmax(logits[0, -1], dim=0)
            pred_id = probs.argmax().item()
            pred = model.tokenizer.decode(pred_id)
            confidence = probs[target_id].item()

            print(
                f"🔭 Layer {layer:2d}:\n"
                f"\tPrediction = {repr(pred):>5}\n"
                f"\tprob('{target_token}') = {confidence:.4f}\n\n"
            )


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = 'gpt2xl'
    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct-Original"

    model = HookedTransformer.from_pretrained(
        model_name, device="cuda" if torch.cuda.is_available() else "mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "The capital of France is"
    target = " Paris"

    tokens = tokenizer(prompt, return_tensors="pt")[
        "input_ids"].to(model.cfg.device)

    # Evaluate using logit lens at selected layers
    num_layers = model.cfg.n_layers
    selected_layers = list(range(num_layers))

    logit_lens_eval(model, tokens, layers=selected_layers, target_token=target)

    # ---------

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-uncased", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    prompt = "What is 5 + 3? 5 + 3 is [MASK]"
    inputs = tokenizer(prompt, return_tensors="pt")
    mask_index = (inputs["input_ids"] ==
                tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     probs = torch.softmax(logits[0, mask_index], dim=-1)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

    # layer_num = 5  # or any layer from 1 to 12
    for layer_num in range(len(hidden_states)):
        masked_hidden = hidden_states[layer_num][0,
                                                mask_index]  # shape: [1, hidden_dim]

        # Apply the MLM head manually
        transform = model.cls
        transformed = transform.predictions.transform(
            masked_hidden)  # Linear + GELU + LayerNorm
        logits = transform.predictions.decoder(transformed)  # vocab logits

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        target_id = tokenizer.convert_tokens_to_ids("8")
        print(f"Layer {layer_num} prob('8'): {probs[0, target_id].item():.8f}")
