import torch
from transformers import AutoTokenizer, XLMRobertaForCausalLM, AutoModelForMaskedLM
from transformers import pipeline, AutoTokenizer, XLMRobertaTokenizer
from optimum.onnxruntime import ORTModelForMaskedLM

print("Done")

tok_dir = "./tokenizer/"
model_dir = "./model/"

print("Done")
# 1. Load the ONNX model
model = ORTModelForMaskedLM.from_pretrained(model_dir)
print("Done")
# 2. Corrected: Load the tokenizer from the correct tokenizer folder
tokenizer = AutoTokenizer.from_pretrained(tok_dir)
print("Done")
inputs = tokenizer("Bonjour, je suis un modèle <mask>.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
