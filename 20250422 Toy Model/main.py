# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "datasets",
#   "matplotlib",
#   "scikit-learn"
# ]
# ///

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
dataset_source = "truthfulqa"  # 'truthfulqa' or 'boolq' or 'both'

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
# ✅ Load dataset
# --------------------

print(f"📄 Loading dataset: {dataset_source.upper()}")

examples = []

if dataset_source in ["truthfulqa", "both"]:
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

if dataset_source in ["boolq", "both"]:
    print("📥 Loading BOOLQ...")
    bq = load_dataset("boolq")["train"]

    for row in bq:
        question = row["question"]
        passage = row["passage"]
        label = 1 if row["answer"] else 0
        examples.append({"text": f"{question} {passage}", "label": label})

    print(f"✅ BOOLQ added. Total examples now: {len(examples)}")

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


def get_hidden_states(examples):
    all_hidden_states = []
    labels = []

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            print(f"    → Processing example {i+1}/{len(examples)}")
            print(f"      ↳ Text: {ex['text']} | Label: {ex['label']}")

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

for layer in range(13):
    train_feats = train_hidden_states[:, layer, :]
    test_feats = test_hidden_states[:, layer, :]
    train_lbls = train_labels
    test_lbls = test_labels

    print(f"\n🧪 Training probe on layer {layer}...")
    probe, loss = train_probe(train_feats, train_lbls)
    print(f"    ↳ Final train loss: {loss:.4f}")

    with torch.no_grad():
        preds = (probe(test_feats) > 0.5).long()
        acc = (preds == test_lbls).float().mean().item()
        accuracies.append(acc)
        probes.append(probe)

    print(f"    ✅ Test accuracy on layer {layer}: {acc:.3f}")

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
