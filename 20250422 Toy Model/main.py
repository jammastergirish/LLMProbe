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

import math
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import random
import transformer_lens
from transformer_lens import HookedTransformer

# --------------------
# ✅ Dataset source
# --------------------
# options: "truthfulqa", "boolq", "truefalse", or "all"
dataset_source = "truefalse"

# --------------------
# ✅ Model selection
# --------------------
model_name = "bert-base-uncased" 
# model_name = "bert-large-uncased" 
# model_name = 'roberta-base'
# model_name = 'gpt2'
# model_name = 'meta-llama/Llama-3.2-1B-Instruct'
# model_name = 'meta-llama/Llama-3.2-1B'
# model_name = 'meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8'
# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "deepseek-ai/DeepSeek-V3-Base"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

use_control_tasks = True

# --------------------
# ✅ Device setup (MPS on Mac, CUDA with GPU, etc.)
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using {device}")


def is_decoder_only_model(model_name):
    decoder_keywords = ["gpt", "llama", "mistral", "pythia", "deepseek"]
    return any(keyword in model_name.lower() for keyword in decoder_keywords)


def get_num_layers(model):
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers + 1
    elif hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        return model.cfg.n_layers
    else:
        raise AttributeError(
            "Cannot determine number of layers for this model")

# --------------------
# ✅ Load model and tokenizer
# --------------------


def load_model_and_tokenizer(model_name):
    use_transformerlens = is_decoder_only_model(model_name)

    if use_transformerlens:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(model_name, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set tokenizer padding if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        if is_decoder_only_model(model_name):
            model_class = AutoModelForCausalLM
        else:
            model_class = AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" if is_decoder_only_model(model_name) else "right"

        model = model_class.from_pretrained(
            model_name, output_hidden_states=True).to(device)
        model.eval()

    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(model_name)
print(f"✅ {model_name.upper()} is ready.")

# --------------------
# ✅ Load dataset
# --------------------


def generate_arithmetic_dataset(n=5000):
    data = []
    while len(data) < n:
        a = random.randint(0, 100)
        b = random.randint(0, 100)

        # 50% chance of being true
        if len(data) % 2 == 0:
            correct_sum = a + b
            text = f"{a} + {b} = {correct_sum}"
            label = 1
        else:
            incorrect_sum = a + b + random.choice(
                [i for i in range(-10, 11) if i != 0])  # avoid correct answer
            text = f"{a} + {b} = {incorrect_sum}"
            label = 0

        data.append({"text": text, "label": label})

    return data

print(f"📄 Loading dataset: {dataset_source.upper()}")

examples = []

if dataset_source in ["truthfulqa", "all"]:
    print("📥 Loading TruthfulQA (multiple_choice)...")
    tq = load_dataset("truthful_qa", "multiple_choice")["validation"]

    for row in tq:
        q = row.get("question", "")
        targets = row.get("mc1_targets", {})
        choices = targets.get("choices", [])
        labels = targets.get("labels", [])
        for answer, label in zip(choices, labels):
            examples.append({"text": f"{q} {answer}", "label": label})

    print(f"✅ TruthfulQA: {len(examples)} total QA-label pairs so far.")

if dataset_source in ["boolq", "all"]:
    print("📥 Loading BOOLQ...")
    bq = load_dataset("boolq")["train"]

    for row in bq:
        question = row["question"]
        passage = row["passage"]
        label = 1 if row["answer"] else 0
        examples.append({"text": f"{question} {passage}", "label": label})

    print(f"✅ BOOLQ added. Total examples now: {len(examples)}")

if dataset_source in ["truefalse", "all"]:
    from datasets import concatenate_datasets

    print("📥 Loading TRUEFALSE (pminervini/true-false)...")
    tf_splits = [
        "animals",
        "cities",
        "companies",
        # "cieacf",
        "inventions",
        "facts",
        "elements",
        "generated"
    ]

    datasets_list = []
    for split in tf_splits:
        split_ds = load_dataset("pminervini/true-false", split=split)
        datasets_list.append(split_ds)

    tf = concatenate_datasets(datasets_list)

    for row in tf:
        examples.append({"text": row["statement"], "label": row["label"]})

    print(f"✅ TRUEFALSE added. Total examples now: {len(examples)}")

if dataset_source in ["arithmetic", "all"]:
    print("📐 Generating ARITHMETIC dataset...")
    arithmetic = generate_arithmetic_dataset(5000)
    examples.extend(arithmetic)
    print(f"✅ Arithmetic dataset added. Total examples now: {len(examples)}")

print(f"✅ Prepared {len(examples)} labeled examples for probing.")

# --------------------
# ✅ Split into train/test sets
# --------------------
print("🧪 Splitting into train/test sets...")
train_examples, test_examples = train_test_split(
    examples, test_size=0.2, random_state=42, shuffle=True
)
print(f"→ Train: {len(train_examples)} examples")
print(f"→ Test: {len(test_examples)} examples")

# --------------------
# ✅ Extract hidden states for train/test separately
# --------------------


def get_hidden_states_hf(examples, return_layer=None):
    all_hidden_states = []
    labels = []

    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"    → Processing example with Hugging Face {i+1}/{len(examples)}")
            print(f"      ↳ Text: {ex['text']} | Label: {ex['label']}")

        inputs = tokenizer(ex['text'], return_tensors='pt',
                           truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

        # Use last token for decoder-only models, first for encoder-only
        if is_decoder_only_model(model_name):
            cls_embeddings = torch.stack(
                [layer[:, -1, :] for layer in hidden_states])
        else:
            cls_embeddings = torch.stack(
                [layer[:, 0, :] for layer in hidden_states])

        # [num_layers, hidden_dim]
        all_hidden_states.append(cls_embeddings.squeeze(1))
        labels.append(ex['label'])

    all_hidden_states = torch.stack(all_hidden_states).to(
        device)  # [N, num_layers, d_model]
    labels = torch.tensor(labels).to(device)

    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], labels  # (N, d_model)
    else:
        return all_hidden_states, labels  # (N, num_layers, d_model)


def get_hidden_states_transformerlens(examples, model, model_name, tokenizer, return_layer=None):
    all_hidden_states = []
    labels = []

    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"    → Processing example with TransformerLens {i+1}/{len(examples)}")
            print(f"      ↳ Text: {ex['text']} | Label: {ex['label']}")

        tokens = tokenizer.encode(
            ex["text"], return_tensors="pt").to(model.cfg.device)

        # Run with cache to extract all activations
        _, cache = model.run_with_cache(tokens)

        # Choose position index (last token for decoder-only, first otherwise)
        pos = -1 if is_decoder_only_model(model_name) else 0

        # Get post-residual activations from each layer
        layer_outputs = [
            cache["resid_post", layer_idx][0, pos, :]  # shape: (d_model,)
            for layer_idx in range(model.cfg.n_layers)
        ]

        # Stack into shape: (num_layers, d_model)
        hidden_stack = torch.stack(layer_outputs)
        all_hidden_states.append(hidden_stack)
        labels.append(ex["label"])

    # Shape: (N, L, D)
    all_hidden_states = torch.stack(all_hidden_states).to(device)
    labels = torch.tensor(labels).to(device)

    # Allow slicing if return_layer is specified
    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], labels  # (N, D)
    else:
        return all_hidden_states, labels  # (N, L, D)


def get_hidden_states(examples, return_layer=None):
    if "HookedTransformer" in str(type(model)):
        return get_hidden_states_transformerlens(examples, model, model_name, tokenizer, return_layer)
    else:
        return get_hidden_states_hf(examples, return_layer)


print("🔍 Extracting embeddings for TRAIN set...")
train_hidden_states, train_labels = get_hidden_states(train_examples)
print("🔍 Extracting embeddings for TEST set...")
test_hidden_states, test_labels = get_hidden_states(test_examples)

# --------------------
# ✅ Define Linear Probe
# --------------------


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

# --------------------
# ✅ Train linear probe on one layer
# --------------------


def train_probe(features, labels, epochs=100, lr=1e-2):
    probe = LinearProbe(features.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(features)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

    return probe, loss.item()


# --------------------
# ✅ Train & evaluate across all BERT layers (on test set)
# --------------------
print("🚀 Training linear probes on TRAIN and evaluating on TEST...")
probes = []
accuracies = []
control_accuracies = []
selectivities = []

for layer in range(get_num_layers(model) + 1):
    train_feats = train_hidden_states[:, layer, :]
    test_feats = test_hidden_states[:, layer, :]
    train_lbls = train_labels
    test_lbls = test_labels

    print(f"\n🧪 Training probe on layer {layer}...")
    probe, loss = train_probe(train_feats, train_lbls)
    probes.append(probe)

    with torch.no_grad():
        preds = (probe(test_feats) > 0.5).long()
        acc = (preds == test_lbls).float().mean().item()
        accuracies.append(acc)
    print(f"    ✅ Test accuracy: {acc:.3f}")

    # 🎲 Control task: shuffle train labels
    if use_control_tasks:
        shuffled_labels = train_lbls[torch.randperm(train_lbls.size(0))]
        ctrl_probe, _ = train_probe(train_feats, shuffled_labels)

        with torch.no_grad():
            ctrl_preds = (ctrl_probe(test_feats) > 0.5).long()
            ctrl_acc = (ctrl_preds == test_lbls).float().mean().item()
            control_accuracies.append(ctrl_acc)

            selectivity = acc - ctrl_acc
            selectivities.append(selectivity)

        print(f"    🎲 Control accuracy: {ctrl_acc:.3f}")
        print(f"    📊 Selectivity: {selectivity:.3f}")

# --------------------
# ✅ Plot Accuracy vs. BERT Layer
# --------------------
print("📊 Plotting accuracy by layer...")
plt.plot(range(len(accuracies)), accuracies, marker='o')
plt.title(
    f"Truth Detection Accuracy per Layer ({model.config.name_or_path})")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{model_name}-{dataset_source}-probe_accuracy_per_layer.png")  # 💾 Save the plot
plt.show()
print("✅ Plot saved")

if use_control_tasks:
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(selectivities)), selectivities,
             marker='o', label="Selectivity")
    plt.title(f"Selectivity per BERT Layer ({model.config.name_or_path})")
    plt.xlabel("Layer")
    plt.ylabel("Selectivity = Real Acc - Control Acc")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("selectivity_by_layer.png")
    plt.show()
    print("✅ Saved selectivity_by_layer.png")


# --------------------
# ✅ PCA and confusion plots for all layers
# --------------------
print("🌀 Generating PCA + confusion matrix plots for all 13 layers...")


cols = math.ceil(math.sqrt(num_layers))
rows = math.ceil(num_layers / cols)

fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
axs = axs.flatten()

for layer in range(get_num_layers(model) + 1):  # 25 for BERT-large
    feats = test_hidden_states[:, layer, :].cpu().numpy()
    lbls = test_labels.cpu().numpy()

    # PCA
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(feats)

    # Probing predictions
    probe = probes[layer]
    with torch.no_grad():
        preds = (probe(torch.tensor(feats).to(device))
                 > 0.5).long().cpu().numpy()

    acc = (preds == lbls).mean()

    ax = axs[layer]
    ax.scatter(feats_2d[lbls == 1][:, 0], feats_2d[lbls == 1]
               [:, 1], color='green', alpha=0.6, label="True", s=10)
    ax.scatter(feats_2d[lbls == 0][:, 0], feats_2d[lbls == 0]
               [:, 1], color='red', alpha=0.6, label="False", s=10)
    ax.set_title(f"Layer {layer} (Acc={acc:.2f})")
    ax.set_xticks([])
    ax.set_yticks([])

    if layer >= len(axs):
        break

plt.tight_layout()
plt.suptitle("PCA of CLS embeddings by BERT Layer", fontsize=16, y=1.02)
plt.savefig(f"{model_name}-{dataset_source}-layerwise_pca_grid.png")
plt.show()
print("✅ Saved as layerwise_pca_grid.png")


# layer_to_visualize = 10  # Try other layers too if you want
# features = test_hidden_states[:, layer_to_visualize, :].cpu().numpy()
# labels_np = test_labels.cpu().numpy()

# print(f"🔍 Running t-SNE on CLS embeddings from layer {layer_to_visualize}...")

# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
# features_2d = tsne.fit_transform(features)

# plt.figure(figsize=(8, 6))
# plt.scatter(features_2d[labels_np == 1][:, 0], features_2d[labels_np == 1]
#             [:, 1], label="True", alpha=0.6, color='green')
# plt.scatter(features_2d[labels_np == 0][:, 0], features_2d[labels_np == 0]
#             [:, 1], label="False", alpha=0.6, color='red')
# plt.legend()
# plt.title(f"t-SNE of CLS Embeddings (Layer {layer_to_visualize})")
# plt.tight_layout()
# plt.savefig(f"{model_name}-{dataset_source}-tsne_layer10.png")
# plt.show()
# print("✅ Saved t-SNE plot as tsne_layer10.png")

print("📈 Generating truth direction projection histograms for all layers...")

rows = cols = math.ceil((get_num_layers(model) + 1) ** 0.5)
fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
axs = axs.flatten()

for layer in range(get_num_layers(model) + 1):
    feats = test_hidden_states[:, layer, :]
    lbls = test_labels

    probe = probes[layer]
    with torch.no_grad():
        projection = torch.matmul(feats, probe.linear.weight[0])  # shape: [N]

    ax = axs[layer]
    ax.hist(projection[lbls == 1].cpu(), bins=30,
            alpha=0.6, label="True", color='green')
    ax.hist(projection[lbls == 0].cpu(), bins=30,
            alpha=0.6, label="False", color='red')
    ax.set_title(f"Layer {layer}")
    ax.set_xticks([])
    ax.set_yticks([])

axs[13].legend()
axs[13].axis('off')
plt.tight_layout()
plt.suptitle("Projection onto Truth Direction per Layer", fontsize=20, y=1.02)
plt.savefig(f"{model_name}-{dataset_source}-truth_projection_grid.png")
plt.show()
print("✅ Saved truth_projection_grid.png")

# ----
# print("🧪 Running causal interventions...")

# layer = 10
# feats = test_hidden_states[:, layer, :]
# lbls = test_labels

# # Find a true and false example
# i_false = (lbls == 0).nonzero(as_tuple=True)[0][0]
# i_true = (lbls == 1).nonzero(as_tuple=True)[0][0]

# x_false = feats[i_false]
# x_true = feats[i_true]

# probe = probes[layer]
# weight_vector = probe.linear.weight[0].detach()

# # Interpolate and project
# alphas = torch.linspace(0, 1, steps=20)
# scores = []
# for alpha in alphas:
#     x_mix = (1 - alpha) * x_false + alpha * x_true
#     projection = torch.dot(x_mix, weight_vector)
#     score = torch.sigmoid(projection).item()
#     scores.append(score)

# # OPTIONAL: Flip direction if projections decrease with alpha
# if scores[-1] < scores[0]:
#     scores = [-s for s in scores]

# # Plot
# plt.figure(figsize=(6, 4))
# plt.plot(alphas.cpu().numpy(), scores)
# plt.xlabel("Truth Injection (alpha)")
# plt.ylabel("Dot Product with Truth Direction")
# plt.title("Interpolated Projection onto Truth Direction")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{model_name}-{dataset_source}-causal_intervention_projection.png")
# plt.show()
# print("✅ Saved causal_intervention_projection.png")

print("✅ Done!")
