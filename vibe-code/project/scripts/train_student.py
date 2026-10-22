#!/usr/bin/env python3
"""Q-031/Q-032 starter trainer: MRI-only student + strict non-gold weak labels.

Requires explicit gold exclusion list, fixed folds.csv, and normalized Glimmer
weak-label CSV. Never reads reports or train.csv gold targets.

Example: python scripts/train_student.py --data-root "$RSNA_DATA" \\
  --weak-labels work/weak_labels.csv --folds work/folds.csv \\
  --gold-ids work/gold_58_uids.txt --backbone-weights work/convnext_tiny_backbone.pth \\
  --out-dir work/q031_run --size 224 --slices 16 --max-series 6 --batch-size 1
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from oxalpha.data.dataset import SeriesRecord, StudyBagDataset, collate_studies
from oxalpha.data.preprocessing import PreprocessConfig, PREPROCESS_VERSION
from oxalpha.models.student import QuartzKneeStudent, StudentConfig, LABELS
from oxalpha.training.records import (load_weak_labels, load_folds, read_gold_ids,
                                      training_partition, file_sha256)
from oxalpha.training.engine import train_epoch, validate_epoch
from oxalpha.training.checkpoints import (save_training_checkpoint, load_training_checkpoint,
                                          canonical_hash)


class SupervisedStudies(Dataset):
    """Supervision is NOT attached to StudyBagDataset and never enters model kwargs."""
    def __init__(self, bags: StudyBagDataset, labels: dict):
        self.bags = bags
        self.labels = labels
        missing = set(bags.study_ids) - set(labels)
        if missing:
            raise ValueError(f'{len(missing)} studies lack weak-label records')

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int):
        inp = self.bags[index]
        uid = inp['study_ids']
        label = self.labels[uid]
        return inp, np.asarray(label.targets, dtype=np.float32), np.asarray(label.confidence, dtype=np.float32)


def supervised_collate(samples: list) -> tuple[dict, np.ndarray, np.ndarray]:
    return (collate_studies([s[0] for s in samples]),
            np.stack([s[1] for s in samples]),
            np.stack([s[2] for s in samples]))


def load_series_csv(data_root: Path) -> list[SeriesRecord]:
    path = data_root / 'train_series.csv'
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        needed = {'StudyInstanceUID', 'SeriesInstanceUID'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f'{path} requires {needed}')
        return [SeriesRecord.from_row(row, data_root / 'train_series') for row in reader]


def make_optimizer(model: QuartzKneeStudent, lr_head: float, lr_backbone: float,
                   weight_decay: float) -> torch.optim.Optimizer:
    named = [(n, p) for n,p in model.named_parameters() if p.requires_grad]
    backbone = [p for n,p in named if n.startswith('encoder.backbone.')]
    head = [p for n,p in named if not n.startswith('encoder.backbone.')]
    if not head:
        raise ValueError('no trainable student head parameters')
    param_groups = [{'params': head, 'lr': lr_head, 'base_lr': lr_head}]
    if backbone:
        param_groups.append({'params': backbone, 'lr': lr_backbone, 'base_lr': lr_backbone})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def make_scheduler(optimizer, *, epochs: int, warmup: int):
    import math
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        total = max(1, epochs - warmup)
        progress = min(1., max(0., (step - warmup) / total))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _write_json_atomic(path: Path, obj: dict):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def _coverage(labels: dict, uids: list[str]) -> dict:
    y = np.asarray([labels[u].targets for u in uids])
    w = np.asarray([labels[u].confidence for u in uids])
    known = np.isfinite(y) & (y >= 0) & (y <= 1) & (w > 0)
    return {label: {'pos': int(np.sum(known[:, i] & (y[:, i] == 1))),
                    'neg': int(np.sum(known[:, i] & (y[:, i] == 0))),
                    'soft': int(np.sum(known[:, i] & (y[:, i] > 0) & (y[:, i] < 1))),
                    'expected_positives': float(np.sum(np.where(known[:, i], y[:, i], 0))),
                    'missing_or_zero_confidence': int(np.sum(~known[:, i]))}
            for i, label in enumerate(LABELS)}


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--weak-labels', required=True, help='Q-020 normalized weak-label CSV or Parquet; not raw reports')
    p.add_argument('--folds', required=True, help='Frozen Q-023 train/val split CSV')
    p.add_argument('--gold-ids', required=True, help='Mandatory nonempty gold-study exclusion UID list')
    p.add_argument('--out-dir', required=True, help='Private scratch/work directory, never public Git')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--slices', type=int, default=16)
    p.add_argument('--max-series', type=int, default=6)
    p.add_argument('--chunk-size', type=int, default=4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--lr-head', type=float, default=5e-4)
    p.add_argument('--lr-backbone', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--backbone-name', default='convnext_tiny.fb_in22k_ft_in1k')
    p.add_argument('--backbone-weights', help='Local trusted timm backbone state_dict')
    p.add_argument('--pretrained', action='store_true', help='Use timm downloads/cached weights (not offline)')
    p.add_argument('--unfreeze-backbone', action='store_true', help='Memory-expensive; Q-033 experiment')
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--resume', help='Existing last.pt from same run & exact config/data')
    p.add_argument('--limit-train', type=int, default=None, help='Only for debugging; changes experiment identity')
    p.add_argument('--limit-val', type=int, default=None, help='Only for debugging; changes experiment identity')
    p.add_argument('--cpu', action='store_true', help='Synthetic/unit tests only; prohibit real large CPU training')
    p.add_argument('--no-amp', action='store_true')
    args = p.parse_args(argv)
    for name in ('epochs', 'batch_size', 'accumulate', 'max_series', 'size', 'slices', 'chunk_size', 'd_model'):
        if getattr(args, name.replace('-', '_')) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.num_workers < 0 or args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        p.error('invalid worker/warmup configuration')
    if args.pretrained and args.backbone_weights:
        p.error('choose either --pretrained or --backbone-weights')
    if not args.pretrained and not args.backbone_weights:
        p.error('Require pretrained backbone: supply --pretrained or --backbone-weights')
    if args.num_workers > 0:
        warnings.warn('DICOM decoding workers may overload shared Slate storage; start with num-workers=0')
    return args


def main(argv=None):
    args = _args(argv)
    data_root = Path(args.data_root).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if out.is_relative_to(PROJECT_ROOT) and not ('work' in out.relative_to(PROJECT_ROOT).parts):
        raise ValueError('training outputs must be in a private work/ directory or external scratch')
    device = torch.device('cpu' if args.cpu else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; allocate a GPU or use --cpu for synthetic tests')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    gold = read_gold_ids(args.gold_ids)
    labels = load_weak_labels(args.weak_labels, gold_ids=gold)
    folds = load_folds(args.folds, gold_ids=gold)
    recs = load_series_csv(data_root)
    study_set = {r.study_uid for r in recs}
    train_ids, val_ids = training_partition(labels, folds, study_set, gold_ids=gold)
    if args.limit_train is not None:
        if args.limit_train < 1:
            raise ValueError('--limit-train must be >=1')
        train_ids = train_ids[:args.limit_train]
    if args.limit_val is not None:
        if args.limit_val < 1:
            raise ValueError('--limit-val must be >=1')
        val_ids = val_ids[:args.limit_val]
    if not train_ids or not val_ids:
        raise ValueError('empty train or val after limiting')
    if set(train_ids) & gold or set(val_ids) & gold or set(train_ids) & set(val_ids):
        raise ValueError('gold/validation leakage')
    train_cover = _coverage(labels, train_ids)
    val_cover = _coverage(labels, val_ids)
    if sum(c['pos'] + c['neg'] + c['soft'] for c in train_cover.values()) == 0:
        raise ValueError('no supervised non-gold training labels')
    if sum(c['pos'] + c['neg'] + c['soft'] for c in val_cover.values()) == 0:
        raise ValueError('no supervised validation labels')
    if not any(f.groups for f in folds.values()):
        warnings.warn('No patient/duplicate group column in folds.csv; study-only grouping is unverified')
    pre_cfg = PreprocessConfig(size=args.size, slices=args.slices)
    train_set = SupervisedStudies(StudyBagDataset([r for r in recs if r.study_uid in set(train_ids)],
                        preprocess_config=pre_cfg, max_series=args.max_series, training=True, seed=args.seed), labels)
    val_set = SupervisedStudies(StudyBagDataset([r for r in recs if r.study_uid in set(val_ids)],
                        preprocess_config=pre_cfg, max_series=args.max_series, training=False, seed=args.seed), labels)
    gen = torch.Generator().manual_seed(args.seed)
    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, collate_fn=supervised_collate,
                          pin_memory=device.type == 'cuda', worker_init_fn=_seed_worker,
                          generator=gen if shuffle else None)
    train_loader, val_loader = loader(train_set, True), loader(val_set, False)
    cfg = StudentConfig(d_model=args.d_model, backbone_name=args.backbone_name,
                       pretrained=args.pretrained, backbone_weights=args.backbone_weights,
                       freeze_backbone=not args.unfreeze_backbone, chunk_size=args.chunk_size,
                       activation_checkpointing=args.activation_checkpointing)
    model = QuartzKneeStudent(cfg).to(device)
    optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    scheduler = make_scheduler(optimizer, epochs=args.epochs, warmup=args.warmup_epochs)
    amp = (device.type == 'cuda' and not args.no_amp)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
        init_scale=1024.0,
        growth_interval=2000,
    )
    # Prevent resume from a different data snapshot or model/preprocessing version.
    cli = vars(args).copy()
    cli.pop('resume')
    for key in ('data_root','weak_labels','folds','gold_ids','out_dir','backbone_weights'):
        cli.pop(key, None)
    provenance = {
        'run_config_hash': canonical_hash(cli), 'model_config': asdict(cfg),
        'train_ids_hash': _hash_ids(train_ids), 'val_ids_hash': _hash_ids(val_ids),
        'weak_labels_sha256': file_sha256(args.weak_labels),
        'folds_sha256': file_sha256(args.folds), 'gold_ids_sha256': file_sha256(args.gold_ids),
        'train_series_sha256': file_sha256(data_root/'train_series.csv'),
        'backbone_weights_sha256': file_sha256(args.backbone_weights) if args.backbone_weights else None,
        'preprocess_version': PREPROCESS_VERSION,
        'torch_version': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'target_semantics': 'soft_bernoulli_bce_v2',
        'gold_excluded': True, 'grouping_columns_present': sorted({g[0] for f in folds.values() for g in f.groups}),
    }
    _write_json_atomic(out/'provenance.json', provenance)
    _write_json_atomic(out/'coverage.json', {'train': train_cover, 'val': val_cover})
    print(f'TRAIN studies={len(train_ids)} VAL studies={len(val_ids)} gold-excluded={len(gold)} device={device}', flush=True)
    start_epoch, best_score, best_rule = 0, None, None
    if args.resume:
        start_epoch, best_score, best_rule = load_training_checkpoint(
            args.resume, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, expected_provenance=provenance)
        print(f'Resumed epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        # Reproduce the same sampler order on resume independent of prior epochs.
        gen.manual_seed(args.seed + epoch)
        train_set.bags.set_epoch(epoch)
        seen_train = []
        tr = train_epoch(model, train_loader, optimizer, device=device,
                         accumulate=args.accumulate, amp=amp, scaler=scaler,
                         grad_clip=args.grad_clip, seen_ids=seen_train)
        if sorted(seen_train) != sorted(train_ids):
            raise ValueError('actual iterated train IDs differ from frozen fold!')
        vl, metrics = validate_epoch(model, val_loader, device=device, amp=amp)
        if vl.loss is None:
            raise ValueError('validation yielded no supervised cells')
        scheduler.step()
        # All 12 defined -> official macro; otherwise best is selected by validation loss.
        if metrics['official_macro_auc'] is not None:
            rule, score = 'official_macro_auc', metrics['official_macro_auc']
        else:
            rule, score = 'negative_val_loss', -vl.loss
        if best_rule is not None and rule != best_rule:
            raise ValueError('best metric selection changed across epochs; check fold support')
        if best_score is None or score > best_score:
            best_score, best_rule = score, rule
            best = True
        else:
            best = False
        summary = {'epoch': epoch, 'train': tr.asdict(), 'val': vl.asdict(),
                   'weak_validation_metrics': metrics, 'best_rule': best_rule,
                   'best_score': best_score, 'model_selection_updated': best}
        _write_json_atomic(out/'metrics.json', summary)
        with (out/'history.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(summary, allow_nan=False) + '\n')
        # Only private work dir; raw per-study outputs and reports are never emitted.
        (out/'train_seen_ids.txt').write_text('\n'.join(sorted(seen_train)) + '\n', encoding='utf8')
        ckpt_kwargs = dict(model=model, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, epoch=epoch, best_score=best_score,
                           best_rule=best_rule, provenance=provenance)
        save_training_checkpoint(out/'last.pt', **ckpt_kwargs)
        if best:
            save_training_checkpoint(out/'best_val.pt', **ckpt_kwargs)
        print(f'epoch={epoch+1}/{args.epochs} train_loss={tr.loss} val_loss={vl.loss:.5f} '
              f'valid_auc_labels={metrics["valid_labels"]}/12 steps={tr.optimizer_steps} '
              f'peak_vram_GiB={tr.peak_vram_gib} best={best}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
