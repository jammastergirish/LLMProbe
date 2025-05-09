import torch
import gc

def estimate_memory_requirements(model, batch_size, seq_length=128):
    """Estimate memory requirements dynamically from the model"""

    # Get model parameters
    if hasattr(model, "config"):
        # For HuggingFace models
        hidden_dim = getattr(model.config, "hidden_size", 0)
        num_layers = getattr(model.config, "num_hidden_layers", 0) + 1
    elif hasattr(model, "cfg"):
        # For TransformerLens models
        hidden_dim = getattr(model.cfg, "d_model", 0)
        num_layers = getattr(model.cfg, "n_layers", 0)
    else:
        return {"param_memory": "Unknown", "activation_memory": "Unknown"}

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())

    # Get precision (default to FP32 if can't determine)
    if next(model.parameters()).dtype == torch.float16:
        precision = 2  # bytes for FP16
    elif next(model.parameters()).dtype == torch.int8:
        precision = 1  # bytes for INT8/quantized
    else:
        precision = 4  # bytes for FP32

    # Calculate memory in GB
    param_memory = (param_count * precision) / (1024**3)

    # Activation memory estimate: batch_size × seq_length × hidden_dim × num_layers × precision
    activation_memory = (batch_size * seq_length *
                         hidden_dim * num_layers * precision) / (1024**3)

    # Get current GPU memory usage if available
    current_memory_usage = "N/A"
    if torch.cuda.is_available():
        try:
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            current_memory_usage = f"{current_memory:.2f} GB"
        except:
            pass

    # Free any temporary tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "param_count": f"{param_count/1e9:.2f}B parameters",
        "param_memory": f"{param_memory:.2f} GB",
        "activation_memory": f"{activation_memory:.2f} GB",
        "precision": f"{precision*8} bit" if precision < 4 else "32 bit",
        "current_usage": current_memory_usage
    }