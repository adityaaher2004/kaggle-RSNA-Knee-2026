from oxalpha.models.student import QuartzKneeStudent, StudentConfig

config = StudentConfig(
    backbone_name="convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
    d_model=256,
    n_labels=12,
    heads=4,
    dropout=0.1,
    freeze_backbone=True,
    chunk_size=4,
    activation_checkpointing=False,
    slice_pool="gated",
    series_pool="gated",
)

model = QuartzKneeStudent(config)
