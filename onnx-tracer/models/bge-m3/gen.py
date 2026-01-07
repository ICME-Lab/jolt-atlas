import torch
from transformers import AutoModel, AutoTokenizer
import onnx
from onnxsim import simplify

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RAW_ONNX = "bge-small.onnx"
SIM_ONNX = "network.onnx"


def main():
    print("Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("Preparing dummy input...")
    dummy = tokenizer(
        "Hello world!",
        return_tensors="pt",
        padding="max_length",
        max_length=16,
        truncation=True
    )

    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    print("Exporting to ONNX (opset=17)...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        RAW_ONNX,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None  # fully static model
    )

    print(f"Exported raw ONNX to: {RAW_ONNX}")

    print("Simplifying ONNX model...")
    model_onnx = onnx.load(RAW_ONNX)
    model_simplified, check = simplify(model_onnx)

    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated!")

    onnx.save(model_simplified, SIM_ONNX)
    print(f"Simplified ONNX saved to: {SIM_ONNX}")

    print("DONE! 🎉")


if __name__ == "__main__":
    main()
