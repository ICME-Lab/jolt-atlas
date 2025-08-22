#!/usr/bin/env python3
# multiclass0: sentiment0-style ops (+ ArgMax), NO hashing
# ONNX ops: Gather -> ReduceSum -> Mul -> Add -> ArgMax
import json, numpy as np, torch, torch.nn as nn

# ----- tiny 10-class toy data (edit as you like) -----
texts = [
    "cheap flights to rome",            # travel
    "box office hits this weekend",     # entertainment
    "quarterly earnings beat guidance", # business
    "university admissions tips",       # education
    "hotel booking refund policy",      # travel
    "new streaming series announced",   # entertainment
    "merger and acquisition news",      # business
    "scholarships and grants",          # education
    "this university announced scholarships",  # education
    "university scholarships and grants",      # education
    "flights refund policy",       # travel
    "rome hotel booking",           # travel
    "flights and hotel"             # travel
]

label_names = ["business","education","travel","entertainment","sports",
               "politics","tech","health","science","other"]   # K = 10
# indices into label_names for each text above:
labels = torch.tensor([2,3,0,1,2,3,0,1,1,1,2,2,2], dtype=torch.long)

PAD = 0
L = 8   # keep ≤32 to respect your tensor cap

# ----- vocab (no hashing) -----
vocab = {}
def tokenize(text: str):
    ids = []
    for tok in text.lower().split():
        if tok not in vocab:
            vocab[tok] = len(vocab) + 1  # 0 = PAD
        ids.append(vocab[tok])
    return ids

seqs = [tokenize(t) for t in texts]
arr = np.full((len(texts), L), PAD, dtype=np.int64)
for i, s in enumerate(seqs):
    n = min(len(s), L)
    arr[i, :n] = s[:n]
X = torch.tensor(arr, dtype=torch.long)  # (B, L)

# sanity: keep embedding rows ≤ 32
assert (len(vocab) + 1) <= 32, f"vocab too large: {len(vocab)+1} rows would exceed 32"

# ----- model: pooled scalar -> per-class Mul/Add -> logits(K) -----
class BagOfTokensNoHash(nn.Module):
    """
    Uses sentiment0 ops only:
      Gather(Embedding V+1 x 1) -> ReduceSum -> Mul/Add(per class) -> ArgMax (export wrapper)
    """
    def __init__(self, vocab_size: int, K: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, 1)  # (V+1, 1)  <= 32 elems
        self.W   = nn.Parameter(torch.ones(K))      # (K,) per-class scale (Mul)
        self.b   = nn.Parameter(torch.zeros(K))     # (K,) per-class bias  (Add)
        nn.init.normal_(self.emb.weight, std=0.1)
        with torch.no_grad():
            self.emb.weight[PAD, 0] = 0.0          # PAD contributes nothing

    def forward(self, x):
        e = self.emb(x)               # (B, L, 1)   [Gather]
        s = e.sum(dim=(1, 2))         # (B,)        [ReduceSum]
        logits = s[:, None] * self.W + self.b   # (B, K) [Mul + Add] via broadcast
        return logits

K = len(label_names)
model = BagOfTokensNoHash(len(vocab), K)

# ----- train -----
opt = torch.optim.Adam(model.parameters(), lr=0.03)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(200):
    logits = model(X)                      # (B, K)
    loss = loss_fn(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 40 == 0:
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean().item()
        print(f"epoch {epoch:03d}  loss {loss.item():.4f}  acc {acc:.2f}")

# ----- save artifacts -----
with open("vocab.json", "w") as f:  json.dump(vocab, f, ensure_ascii=False, indent=2)
with open("labels.json", "w") as f: json.dump(label_names, f, ensure_ascii=False, indent=2)
with open("meta.json", "w") as f:   json.dump({"PAD": PAD, "L": L}, f, ensure_ascii=False, indent=2)

# ----- export ONNX: single output = class_id via ArgMax -----
class ExportArgMaxOnly(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner.eval()
    def forward(self, x):
        logits = self.inner(x)               # (1, K)
        return torch.argmax(logits, dim=1)   # ArgMax(axis=1) -> (1,)

dummy = torch.randint(0, len(vocab)+1, (1, L), dtype=torch.long)  # includes PAD=0
torch.onnx.export(
    ExportArgMaxOnly(model).eval(), dummy, "network.onnx",
    input_names=["tokens"], output_names=["class_id"],
    opset_version=15  # fixed (1, L)
)

print(f"Exported network.onnx (class_id); vocab rows={len(vocab)+1} (<=32), L={L}, K={K}")