# Q-030 — QuartzKnee-HMIL model patch

Implementation target: `SPEC.md` Q-ARCH-01/02/03/04 and `TODO.md` Q-030.
This is an **incremental source patch** for your `quartz` worktree, not a Git commit.

## Files

- `src/oxalpha/models/backbone.py`: one shared `timm` ConvNeXt-Tiny encoder, valid-slice filtering, chunking, frozen or trainable mode, optional checkpointing.
- `src/oxalpha/models/mil.py`: class-query gated slice and series MIL with safe all-masked softmax; optional mean/max ablation.
- `src/oxalpha/models/series_context.py`: independently embedded plane/fluid/fat, geometry and missingness, 1-layer/4-head series transformer.
- `src/oxalpha/models/student.py`: complete MRI-only model, 12 logits in spec order, input/mask validation, NumPy-to-Torch collate adapter.
- `src/oxalpha/models/__init__.py`: model exports.
- `tests/test_model_shapes.py`, `tests/test_masking.py`, `tests/test_no_text_leak.py`.

## Install and test

Extract ZIP preserving `src/` and `tests/` structure, in the existing project root. The tests use a tiny injected backbone so they run on login/CPU nodes without timm or GPU access.

```bash
module load python/gpu/3.12.5
python -m pytest -q tests/test_model_shapes.py tests/test_masking.py tests/test_no_text_leak.py
```

**Result in the packaging environment:** 18 passed (CPU, torch 2.10). Production timm construction and actual ConvNeXt weights have **not** been exercised in this environment; timm is not installed here.

For production on Quartz, install a compatible `timm`/`torch` version and **verify** that your selected checkpoint name actually exists. For off-network initialization use a trusted local timm-backbone-only state dict, or load the trained student checkpoint in a later trainer/inference script.

```bash
python -m pip install timm
python - <<'PY'
import timm
models = timm.list_models('convnext_tiny*', pretrained=True)
print('\n'.join(models))
PY
```

## Dataset interface / forward example

```python
import torch
from oxalpha.models.student import QuartzKneeStudent, StudentConfig, prepare_student_batch

# batch = collate_studies([study_a.inference_inputs(), study_b.inference_inputs()])
# NumPy arrays -> correctly typed torch tensors on the model device.
batch = prepare_student_batch(batch, device='cuda')
model = QuartzKneeStudent(StudentConfig(
    pretrained=True,         # for training ONLY when timm can access exact weights
    backbone_name='convnext_tiny.fb_in22k_ft_in1k',
    freeze_backbone=True,
    chunk_size=8,
)).cuda().eval()
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
    logits = model(**batch)              # [B,12]
    probabilities = logits.sigmoid()    # [B,12]
```

`study_ids` is allowed for log parity but not used as a network feature. Model outputs **logits**, not probabilities. The 12 columns, in order, are `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.

`prepare_student_batch` rejects any extra fields including labels, reports, and teacher logits. Training supervision (`y`, `w`) must be passed separately to Q-031's loss, never to `model(**batch)`.

### Memory strategy

**A study's float32 input alone is ~182 MiB** at `[8,24,3,288,288]`, not including backbone intermediates. On V100:

- Start frozen backbone and `chunk_size=4` or `8`, study microbatch `B=1`, input size 224/256 if necessary.
- Frozen mode sets backbone eval and no-grad, retaining gradient flow through new projection/MIL/classifier.
- For later unfreezing, set `model.set_backbone_trainable(True)` and configure `activation_checkpointing=True` when creating the encoder; chunking **without checkpointing does not eliminate retained backward activations**.
- Moving the entire study tensor to GPU still costs its full input memory. Future Q-031 training may use CPU-to-GPU chunk transfer or cached frozen features for better VRAM efficiency.
- Benchmark on allocated GPU hardware before choosing B/K/S/size; do not infer runtime from unit tests.

### Edge-case policies

- Invalid/padded images (including NaNs in invalid slots) are **not passed through the backbone**.
- All-invalid study raises `InvalidStudyError`; Q-040 must handle it with a documented submission fallback, not a NaN prediction.
- Unknown plane=3; unknown fluid/fat=2. Fluid and fat are independently embedded.
- `return_attention=True` returns debug `{'slice': [B,S,K,12], 'series': [B,S,12]}` for gated pooling. Attention indicates prioritization, **not anatomical localization proof**.
- `slice_pool` and `series_pool` support `'mean'` and `'max'` as required ablation controls; attention values become `None`.
- Pretrained ConvNeXt weights are not distributed in this patch and are not verified; training performance/AUC is unmeasured.

## Follow-up

**Q-031**: masked BCE with missing-label/confidence handling; AMP, gradient accumulation, train/validation loop, provenance, checkpoints, and monitoring. **Q-040**: offline submission inference.
