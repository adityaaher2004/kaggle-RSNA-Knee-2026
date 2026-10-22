
#!/usr/bin/env python3
"""Q-017 -> Q-030 real MRI forward-pass smoke test."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_studies import csv_records, validate_study
from oxalpha.data.dataset import preprocess_study, collate_studies
from oxalpha.data.preprocessing import PreprocessConfig
from oxalpha.models.student import (
    QuartzKneeStudent,
    StudentConfig,
    prepare_student_batch,
    LABELS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--study-uid", default=None)
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--slices", type=int, default=24)
    parser.add_argument("--max-series", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run inside a GPU allocation.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    groups = csv_records(Path(args.data_root), args.split)
    candidates = (
        [args.study_uid] if args.study_uid else sorted(groups)
    )

    cfg = PreprocessConfig(
        size=args.size,
        slices=args.slices,
    )

    # Load the first successfully decoded study.
    study = None
    start = time.perf_counter()

    for uid in candidates:
        if uid not in groups:
            raise KeyError(f"Unknown study UID: {uid}")
        try:
            candidate = preprocess_study(
                uid,
                groups[uid],
                preprocess_config=cfg,
                max_series=args.max_series,
                training=False,
            )

            sample = candidate.inference_inputs()
            validate_study(
                sample,
                cfg=cfg,
                max_series=args.max_series,
            )
            study = candidate
            break
        except Exception as exc:
            print(f"Skipping {uid}: {type(exc).__name__}: {exc}")
            if args.study_uid:
                raise

    if study is None:
        raise RuntimeError("No usable study found")

    preprocessing_seconds = time.perf_counter() - start

    print("\nStudy:", study.study_ids)
    print("Selected series:", study.selected_series_uids)
    print("Failed series:", study.failures)
    print("Valid series:", int(study.series_mask.sum()))
    print("Valid slices:", int(study.slice_mask.sum()))
    print("Preprocessing seconds:", round(preprocessing_seconds, 2))

    batch = collate_studies([study.inference_inputs()])
    batch = prepare_student_batch(batch, device=device)

    print("Input shape:", tuple(batch["images"].shape))

    model_cfg = StudentConfig(
        backbone_name="convnext_tiny.fb_in22k_ft_in1k",
        pretrained=not args.no_pretrained and args.weights is None,
        backbone_weights=args.weights,
        d_model=256,
        freeze_backbone=True,
        chunk_size=args.chunk_size,
    )

    model = QuartzKneeStudent(model_cfg).to(device).eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**batch)
            probabilities = torch.sigmoid(logits)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()

    print("\nForward pass successful")
    print("Logits shape:", tuple(logits.shape))
    print("Inference seconds:", round(elapsed, 3))
    print(
        "Peak allocated GPU memory (GiB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    )

    print("\nUntrained model predictions:")
    for name, value in zip(
        LABELS, probabilities[0].float().cpu().tolist()
    ):
        print(f"{name:20s} {value:.5f}")

    print("\nPASS: Real MRI -> 12 finite logits")


if __name__ == "__main__":
    main()

