#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


PLACE_PHRASES = [
    "in a forest",
    "in a meadow",
    "in a quiet valley",
    "in a sunny clearing",
    "in a small grove",
    "in a green field",
    "by a river",
    "by a pond",
    "by a lake",
    "near an old bridge",
    "near a hill",
    "near a stone wall",
    "under a tall tree",
    "under a wooden bridge",
    "beside a narrow path",
    "beside a little stream",
]

ADJECTIVES = [
    "quiet",
    "careful",
    "small",
    "little",
    "gentle",
    "shy",
    "curious",
    "brave",
]

ANIMALS = [
    "fox",
    "rabbit",
    "deer",
    "frog",
    "mouse",
    "bird",
    "duck",
    "squirrel",
]


def build_prompts(count: int, seed: int) -> list[dict]:
    prompts = [
        f"Once upon a time, {place}, a {adj} {animal}"
        for place in PLACE_PHRASES
        for adj in ADJECTIVES
        for animal in ANIMALS
    ]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return [
        {
            "id": i,
            "seed": seed + i,
            "prompt": prompt,
        }
        for i, prompt in enumerate(prompts[:count])
    ]


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="atlas-onnx-tracer/models/qwen",
        help="Local HF model directory.",
    )
    parser.add_argument(
        "--output",
        default="qwen2-awy/calibration/stories_128.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = device_name()
    model_path = Path(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model.eval()

    samples = build_prompts(args.count, args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            prompts = [item["prompt"] for item in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for item, text in zip(batch, texts):
                record = {
                    **item,
                    "text": text,
                    "prompt_tokens": len(tokenizer.encode(item["prompt"])),
                    "tokens": len(tokenizer.encode(text)),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{record['id']:03d}: {record['prompt']} -> {record['tokens']} tokens")


if __name__ == "__main__":
    main()
