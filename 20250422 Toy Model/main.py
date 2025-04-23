# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "datasets",
#   "matplotlib",
#   "scikit-learn"
# ]
# ///

import math
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# --------------------
# ✅ Choose dataset source: 'truthfulqa', 'boolq', or 'both'
# --------------------
dataset_source = "truefalse"  # options: "truthfulqa", "boolq", "truefalse", or "all"

# ✅ Model config
bert_model_name = "bert-large-uncased"  # or "bert-base-uncased"

use_control_tasks = True

# --------------------
# ✅ Device setup (MPS on Mac)
# --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# --------------------
# ✅ Load BERT
# --------------------
print("📦 Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertModel.from_pretrained(
    bert_model_name, output_hidden_states=True).to(device)
model.eval()
print("✅ BERT is ready.")

# --------------------
# ✅ Load dataset
# --------------------

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
        # "animals",
        "cities",
        # "companies",
        # "cieacf",
        "inventions",
        "facts",
        # "elements",
        # "generated"
    ]

    datasets_list = []
    for split in tf_splits:
        split_ds = load_dataset("pminervini/true-false", split=split)
        datasets_list.append(split_ds)

    tf = concatenate_datasets(datasets_list)

    for row in tf:
        examples.append({"text": row["statement"], "label": row["label"]})

    print(f"✅ TRUEFALSE added. Total examples now: {len(examples)}")

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


def get_hidden_states(examples, return_layer=None):
    all_hidden_states = []
    labels = []

    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"    → Processing example {i+1}/{len(examples)}")
            print(f"      ↳ Text: {ex['text']} | Label: {ex['label']}")

        inputs = tokenizer(ex['text'], return_tensors='pt',
                           truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

        cls_embeddings = torch.stack([layer[:, 0, :]
                                     # [13, 1, 768]
                                      for layer in hidden_states])
        all_hidden_states.append(cls_embeddings.squeeze(1))  # [13, 768]
        labels.append(ex['label'])

    all_hidden_states = torch.stack(all_hidden_states).to(device)
    labels = torch.tensor(labels).to(device)

    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], labels  # (N, 768), (N,)
    else:
        return all_hidden_states, labels  # (N, 13, 768), (N,)


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

for layer in range(model.config.num_hidden_layers + 1):
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
    f"Truth Detection Accuracy per BERT Layer ({model.config.name_or_path})")
plt.xlabel("BERT Layer (0=Embedding, 1-25=Transformer Layers)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("output.png")  # 💾 Save the plot
plt.show()
print("✅ Plot saved as output.png")

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


num_layers = model.config.num_hidden_layers + 1  # e.g. 25 for BERT-large
cols = math.ceil(math.sqrt(num_layers))
rows = math.ceil(num_layers / cols)

fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
axs = axs.flatten()

for layer in range(num_layers):  # 25 for BERT-large
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
plt.savefig("layerwise_pca_grid.png")
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
# plt.savefig("tsne_layer10.png")
# plt.show()
# print("✅ Saved t-SNE plot as tsne_layer10.png")

print("📈 Generating truth direction projection histograms for all layers...")

rows = cols = math.ceil((model.config.num_hidden_layers + 1) ** 0.5)
fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
axs = axs.flatten()

for layer in range(model.config.num_hidden_layers + 1):
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
plt.savefig("truth_projection_grid.png")
plt.show()
print("✅ Saved truth_projection_grid.png")

# ----
print("🧪 Running causal interventions...")

layer = 10
feats = test_hidden_states[:, layer, :]
lbls = test_labels

# Find a true and false example
i_false = (lbls == 0).nonzero(as_tuple=True)[0][0]
i_true = (lbls == 1).nonzero(as_tuple=True)[0][0]

x_false = feats[i_false]
x_true = feats[i_true]

probe = probes[layer]
weight_vector = probe.linear.weight[0].detach()

# Interpolate and project
alphas = torch.linspace(0, 1, steps=20)
scores = []
for alpha in alphas:
    x_mix = (1 - alpha) * x_false + alpha * x_true
    projection = torch.dot(x_mix, weight_vector)
    score = torch.sigmoid(projection).item()
    scores.append(score)

# OPTIONAL: Flip direction if projections decrease with alpha
if scores[-1] < scores[0]:
    scores = [-s for s in scores]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(alphas.cpu().numpy(), scores)
plt.xlabel("Truth Injection (alpha)")
plt.ylabel("Dot Product with Truth Direction")
plt.title("Interpolated Projection onto Truth Direction")
plt.grid(True)
plt.tight_layout()
plt.savefig("causal_intervention_projection.png")
plt.show()
print("✅ Saved causal_intervention_projection.png")

print("✅ Done!")
