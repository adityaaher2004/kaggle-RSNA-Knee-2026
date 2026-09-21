import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
import numpy as np
import onnxruntime as ort

tok_dir = "./tokenizer/"
tokenizer = AutoTokenizer.from_pretrained(tok_dir)


# Prepare input
inputs = tokenizer("Bonjour, je suis un modèle <mask>.",
                   return_tensors="np")

session = ort.InferenceSession("model.onnx")
onnx_inputs = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64)
}

# 5. Run ONNX Inference
onnx_outputs = session.run(['outputs'], onnx_inputs)
print(onnx_outputs)
embeddings = onnx_outputs[0]

# 6. Extract and align tokens AFTER inference
input_ids = inputs["input_ids"][0]  # Get the array for the first sentence
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# 7. Print the mapping
print(f"Sequence length: {len(tokens)} tokens.\n")
for i, token in enumerate(tokens):
    # Extract the hidden vector specific to this token index
    token_vector = embeddings[0][i]

    # Preview token alongside the first 3 dimensions of its vector
    print(f"Token: {token:<12} | Vector preview: {token_vector[:3]}")






