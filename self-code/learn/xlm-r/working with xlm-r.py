import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",  # or "fp4" for float 4-bit quantization
    bnb_4bit_use_double_quant=True,  # use double quantization for better performance
)
tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-roberta-large")
model = AutoModelForMaskedLM.from_pretrained(
    "facebook/xlm-roberta-large",
    device_map="auto",
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config
)

inputs = tokenizer("Bonjour, je suis un modèle <mask>.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))