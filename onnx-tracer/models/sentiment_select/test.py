#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODEL_PATH = str(HERE / "network.onnx")   # exported model (one output: label_bool)
VOCAB_PATH = HERE / "vocab.json"          # saved during training
L = 5                                     # must match training/export
PAD_ID = 0

# --- load vocab ---
with VOCAB_PATH.open() as f:
    VOCAB = json.load(f)

def tok(text: str):
    return [VOCAB.get(t, PAD_ID) for t in text.lower().split()]

def pad1(ids, L):
    arr = np.full((1, L), PAD_ID, dtype=np.int64)
    n = min(len(ids), L)
    arr[0, :n] = ids[:n]
    return arr

def run_one(sess, text: str):
    ids = tok(text)
    x = pad1(ids, L)                           # shape (1, L), int64
    out = sess.run(None, {"tokens": x})[0]     # single output: (1,1) bool
    y_bool = bool(out.ravel()[0])
    y_int = int(y_bool)
    print(f"{text!r}")
    print(f"  tokens -> {x.flatten().tolist()}")
    print(f"  label_bool -> {y_bool}  (as int: {y_int})\n")
    return y_int

if __name__ == "__main__":
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    # Same sample set you trained on; tweak as you like
    texts = [
        "I love this", "This is great", "Happy with the result",
        "I hate this", "This is bad", "Not satisfied",
    ]

    preds = [run_one(sess, t) for t in texts]
    expected = np.array([1,1,1,0,0,0], dtype=np.int64)  # optional, from your earlier toy set
    acc = float((np.array(preds, dtype=np.int64) == expected).mean())
    print(f"acc: {acc:.2f}")