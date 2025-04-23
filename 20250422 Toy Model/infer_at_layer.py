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


def infer_from_any_layer(model: HookedTransformer, tokens: torch.Tensor, layer: int, pos: int = -1):
    """
    Infer logits from a given residual stream starting at any layer.
    
    Args:
        model: HookedTransformer model (from TransformerLens).
        tokens: Input tokens, shape [1, seq_len].
        layer: The layer at which to resume forward pass.
        pos: Token position (default: -1 for final token).
        
    Returns:
        Logits from that point forward.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

        # Extract the residual stream at the given layer
        resid = cache["resid_post", layer]  # shape: [1, seq_len, d_model]

        # Forward through remaining blocks
        x = resid
        for i in range(layer, model.cfg.n_layers):
            x = model.blocks[i](x)

        # Final layernorm + unembed
        x = model.ln_final(x)
        logits = model.unembed(x)
        return logits


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = HookedTransformer.from_pretrained(
        model_name, device="cuda" if torch.cuda.is_available() else "mps")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "What is 5 + 3? "
    tokens = tokenizer(prompt, return_tensors="pt")[
        "input_ids"].to(model.cfg.device)

    # Run inference from any layer (e.g., 10 or final)
    # layer = model.cfg.n_layers - 1
    layer = 10
    logits = infer_from_any_layer(model, tokens, layer=layer)

    # next_token_id = logits[0, -1].argmax().item()
    # prediction = tokenizer.decode(next_token_id)
    # print(f"Token ID: {next_token_id}")
    # print(f"Prediction at layer {layer}: {repr(prediction)}")

    # probs = torch.softmax(logits[0, -1], dim=0)
    # topk = torch.topk(probs, k=10)
    # print("Top predictions at layer", layer)
    # for i in range(10):
    #     token_id = topk.indices[i].item()
    #     print(
    #         f"{i+1}. {repr(tokenizer.decode(token_id))} — {topk.values[i].item():.4f}")


    # Get probability distribution over the vocabulary
    probs = torch.softmax(logits[0, -1], dim=0)

    # Check confidence for all digits 0 through 10
    results = []
    for i in range(11):
        target_str = str(i)
        target_id = tokenizer.encode(target_str, add_special_tokens=False)[0]
        confidence = probs[target_id].item()
        results.append((target_str, confidence))

    # Sort and display
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"🔢 Model confidence in each digit (top predictions):")
    for digit, score in results:
        print(f"  {digit}: {score:.6f}")
