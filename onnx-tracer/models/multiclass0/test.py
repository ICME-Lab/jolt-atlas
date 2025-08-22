#!/usr/bin/env python3
import json, numpy as np
from pathlib import Path
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
LABELS = json.load((HERE/"labels.json").open())
META   = json.load((HERE/"meta.json").open())
VOCAB  = json.load((HERE/"vocab.json").open())
PAD, L = META["PAD"], META["L"]

def tok(text: str):
    """Map tokens to vocab IDs, unknown -> PAD=0"""
    ids = []
    for t in text.lower().split():
        ids.append(VOCAB.get(t, PAD))
    return ids

def pad1(ids, L):
    a = np.full((1, L), PAD, dtype=np.int64)
    a[0, :min(len(ids), L)] = ids[:L]
    return a

def run(model_path, text):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    x = pad1(tok(text), L)
    out = sess.run(None, {"tokens": x})[0]
    if out.ndim == 1:  # class_id onnx (ArgMax export)
        cls = int(out[0])
    else:              # logits onnx
        logits = np.asarray(out).reshape(-1)
        cls = int(np.argmax(logits))
    print(f"{text!r}\n  tokens -> {x.flatten().tolist()}\n  class  -> {cls}: {LABELS[cls]}\n")
    return cls

if __name__ == "__main__":
    model_path = HERE/"network.onnx"
    if not model_path.exists():
        model_path = HERE/"multiclass_logits.onnx"
    print(f"Using model: {model_path.name}")

    texts = [
        "cheap flights to rome",
        "box office hits this weekend",
        "quarterly earnings beat guidance",
        "university admissions tips",
        "new streaming series announced",
        "this university announced scholarships",
        "merger and acquisition news",
        "scholarships news",
        "university scholarships news",
        "this university grants scholarships",
    ]
    preds = [run(model_path, t) for t in texts]