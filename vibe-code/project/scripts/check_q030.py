import torch
from oxalpha.models.student import QuartzKneeStudent, StudentConfig

config = StudentConfig(
    backbone_name="convnext_tiny.fb_in22k_ft_in1k",
    pretrained=False,
    d_model=256,
    freeze_backbone=True,
    chunk_size=4,
    backbone_weights="convnext_tiny_backbone.pth"
)

model = QuartzKneeStudent(config)

print("Model initialized successfully")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")

if torch.cuda.is_available():
    model = model.cuda()
    print("GPU:", torch.cuda.get_device_name(0))
    print("Allocated GPU memory:", round(torch.cuda.memory_allocated() / 1024**3, 3), "GiB")
else:
    print("CUDA unavailable; model remains on CPU")
