import torch
import timm

backbone = timm.create_model(
    "convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
    num_classes=0,
    global_pool="avg",
)

torch.save(
    backbone.state_dict(),
    "convnext_tiny_backbone.pth",
)
