# gen_simple_where.py
#!/usr/bin/env python3
import json, numpy as np, torch, torch.nn as nn

texts = [
    "I love this", "This is great", "Happy with the result",
    "I hate this", "This is bad", "Not satisfied"
]
labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)

# ----- vocab -----
vocab = {}
def tokenize(t):
    ids=[]
    for tok in t.lower().split():
        if tok not in vocab: vocab[tok] = len(vocab)+1
        ids.append(vocab[tok])
    return ids

L = 5
arr = np.zeros((len(texts), L), dtype=np.int64)
for i, s in enumerate([tokenize(t) for t in texts]):
    arr[i, :min(len(s), L)] = s[:L]
X = torch.tensor(arr, dtype=torch.long)

class TinyWhereSimple(nn.Module):
    def __init__(self, vocab_size, L, tau=0.5, t=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, 1)     # (B,L,1)
        self.w = nn.Parameter(torch.zeros(1))          # scalar
        self.b = nn.Parameter(torch.zeros(1))          # scalar
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("thresh_t", torch.tensor(float(t)))  # <-- scalar

    def reset_pad(self):
        with torch.no_grad():
            self.emb.weight[0, 0] = 0.0

    def forward(self, x):
        e = self.emb(x)                                     # (B,L,1)
        cond = e >= self.tau                                 # (B,L,1) bool
        masked = torch.where(cond, e, torch.zeros_like(e))  # (B,L,1)

        # Make training logit 1-D: (B,)
        sum_score = masked.sum(dim=(1, 2), keepdim=False)   # (B,)
        logit = sum_score * self.w + self.b                 # (B,)

        # For the boolean output keep (B,1)
        label_bool = (logit >= self.thresh_t).unsqueeze(1)  # (B,1)
        return logit, label_bool

model = TinyWhereSimple(len(vocab), L, tau=0.5, t=0.0)
model.reset_pad()

# ----- train -----
opt = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCEWithLogitsLoss()
for epoch in range(60):
    logit, _ = model(X)                     # logit shape (B,)
    loss = loss_fn(logit, labels)           # labels shape (B,)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 15 == 0:
        with torch.no_grad():
            _, yb = model(X)                # yb is (B,1) bool
            acc = (yb.squeeze(1).long() == labels.long()).float().mean().item()
        print(f"epoch {epoch:02d}  loss {loss.item():.4f}  acc {acc:.2f}")

# ----- save vocab -----
with open("vocab.json", "w") as f: json.dump(vocab, f, ensure_ascii=False, indent=2)

# ----- export: ONE OUTPUT (bool), fixed (1,L) -----
class ExportOnlyLabel(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner.eval()
    def forward(self, x):
        _, y = self.inner(x)
        return y

dummy = torch.randint(1, len(vocab)+1, (1, L), dtype=torch.long)
torch.onnx.export(
    ExportOnlyLabel(model), dummy, "network.onnx",
    input_names=["tokens"], output_names=["label_bool"],
    opset_version=15
)
print("Exported network.onnx (Gather → Where → ReduceSum) and vocab.json")