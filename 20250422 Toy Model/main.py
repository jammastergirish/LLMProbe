# /// script
# dependencies = [
#   "torch",
#  "transformers",
#   "datasets",
# "matplotlib"
# ]
# ///

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# --------------------
# ✅ Device setup (MPS on Mac)
# --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# --------------------
# ✅ Load BERT
# --------------------
print("📦 Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased', output_hidden_states=True).to(device)
model.eval()
print("✅ BERT is ready.")

# --------------------
# ✅ Load TruthfulQA (Multiple Choice)
# --------------------
print("📄 Loading TruthfulQA (multiple_choice)...")
dataset = load_dataset("truthful_qa", "multiple_choice")
data = dataset["validation"]  # or use "train"
print(f"✅ Loaded {len(data)} entries from TruthfulQA.")

# --------------------
# ✅ Format into [{text, label}], label = 1 for correct, 0 for incorrect answers
# --------------------
print("🔄 Processing question/answer pairs...")
examples = []

for row in data:
    if "question" not in row or "mc1_targets" not in row:
        continue

    q = row["question"]
    target = row["mc1_targets"]
    choices = target.get("choices", [])
    labels = target.get("labels", [])

    if not choices or not labels:
        continue

    for answer, label in zip(choices, labels):
        examples.append({"text": f"{q} {answer}", "label": label})

print(f"✅ Prepared {len(examples)} labeled examples for probing.")

# Optional: trim for speed
# examples = examples[:200]  # Try 1000+ when ready

# --------------------
# ✅ Extract hidden states for each example
# --------------------
print("🔍 Extracting [CLS] embeddings from each BERT layer...")


def get_hidden_states(examples):
    all_hidden_states = []
    labels = []

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            print(f"    → Processing example {i+1}/{len(examples)}")
            print(f"      ↳ Text: {ex['text']}")

        inputs = tokenizer(ex['text'], return_tensors='pt',
                           truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of 13 layers

        cls_embeddings = torch.stack([layer[:, 0, :]
                                     # [13, 1, 768]
                                      for layer in hidden_states])
        all_hidden_states.append(cls_embeddings.squeeze(1))  # [13, 768]
        labels.append(ex['label'])

    all_hidden_states = torch.stack(all_hidden_states).to(device)
    labels = torch.tensor(labels).to(device)
    print(
        f"✅ Shape of hidden_states_tensor: {all_hidden_states.shape} (examples, layers, 768)")
    return all_hidden_states, labels


hidden_states_tensor, labels_tensor = get_hidden_states(examples)

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
# ✅ Train & evaluate across all BERT layers
# --------------------
print("🚀 Training linear probes layer by layer...")
probes = []
accuracies = []

for layer in range(13):
    feats = hidden_states_tensor[:, layer, :]  # [N, 768]
    lbls = labels_tensor

    print(f"\n🧪 Training probe on layer {layer}...")
    probe, loss = train_probe(feats, lbls)
    print(f"    ↳ Final loss: {loss:.4f}")

    with torch.no_grad():
        preds = (probe(feats) > 0.5).long()
        acc = (preds == lbls).float().mean().item()
        accuracies.append(acc)
        probes.append(probe)

    print(f"    ✅ Accuracy on layer {layer}: {acc:.3f}")

# --------------------
# ✅ Plot Accuracy vs. BERT Layer
# --------------------
print("📊 Plotting accuracy by layer...")
plt.plot(range(13), accuracies, marker='o')
plt.title("Truth Detection Accuracy per BERT Layer (TruthfulQA)")
plt.xlabel("BERT Layer (0=Embedding, 1-12=Transformer Layers)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("output.png")  # 💾 Save the plot
plt.show()
print("✅ Plot saved as output.png")

print("✅ Done!")
