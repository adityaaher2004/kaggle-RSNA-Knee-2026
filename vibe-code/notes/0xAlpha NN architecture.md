                        TRAINING ONLY
                ┌─────────────────────────────────┐
                │ Radiology Report (multilingual) │
                └───────────┬─────────────────────┘
                            ↓
                Language detector → XLM-R-base
                            ↓
                Structured finding extractor (12×{−1,0,+1} + z_text ∈ R^768)
                            ↓
        ┌───────────────────┴───────────────┐
        │ MULTIMODAL TEACHER (train only)   │
        │ [z_img ‖ z_text] → MLP → τ logits │
        │ + InfoNCE(z_img, z_text)          │
        └───────────────────┬───────────────┘
                            │ logit KD (τ=3), feature KD,
                            │ attention KD, ITC
MRI study                   ↓
 ├── m series (plane, fluid, fat-sup metadata M_j)
 │    ├── N_j slices (sorted via ImagePositionPatient)
 │    │     2.5D 3-channel stacks → ConvNeXt-T/EffNet-B0
 │    │     + 3D-position embedding          [B·m × N × 512]
 │    │     ↓
 │    │   Gated attention MIL (metadata-conditioned query)
 │    │     + 6 pathology-group attention heads   → z_j ∈ R^512
 │    └── (repeat)
 │
 ├── 2-layer transformer over {z_1..z_m} (+ modality tokens)
 ├── metadata-conditioned series attention pooling
 ↓
 z_study ∈ R^512
 ↓
 Label-embedding head (512→64→12)
 ↓
 12 sigmoid logits

