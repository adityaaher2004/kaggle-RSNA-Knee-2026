# AUTO-GENERATED; DO NOT EDIT. Source: specs/contracts.yaml SHA256=834a425dd5249ff2e3700eebd88a3542bb0c1a28010f8f9eb5a48ab45c4d462f
from __future__ import annotations
from typing import TypedDict, Any
SPEC_VERSION = '1.0.0'
MODEL_NAME = 'QuartzKnee-HMIL'
CONTRACT_SHA256 = '834a425dd5249ff2e3700eebd88a3542bb0c1a28010f8f9eb5a48ab45c4d462f'
LABELS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
SUBMISSION_COLUMNS = ('StudyInstanceUID',) + LABELS
PLANES = ('Sagittal', 'Coronal', 'Axial', 'Unknown')
UNKNOWN_METADATA_VALUE = 2
INFERENCE_FIELDS = ('images', 'slice_mask', 'series_mask', 'plane', 'fluid', 'fat', 'geometry_features', 'geometry_mask', 'study_ids')
FORBIDDEN_INFERENCE_FIELDS = ('Report', 'report', 'text', 'text_embedding', 'labels', 'y', 'weak_labels', 'gold_labels', 'teacher_logits')
TENSOR_SHAPES = {'images': '[B,S,K,3,H,W]', 'slice_mask': '[B,S,K]', 'series_mask': '[B,S]', 'plane': '[B,S]', 'fluid': '[B,S]', 'fat': '[B,S]', 'geometry_features': '[B,S,G]', 'geometry_mask': '[B,S,G]', 'logits': '[B,12]', 'submission': '[N,13]'}
CONFIG_DEFAULTS = {'data': {'max_series': 8, 'slices_per_series': 24, 'image_size': 288, 'percentile_low': 0.5, 'percentile_high': 99.5, 'channel_mode': '2p5d'}, 'model': {'family': 'convnext_tiny', 'd_model': 256, 'transformer_heads': 4, 'transformer_layers': 1}, 'train': {'seed': 42, 'head_lr': 0.0005, 'backbone_lr': 3e-05, 'weight_decay': 0.05}, 'gold': {'use_for_training': False}}

class StudentBatch(TypedDict):
    images: Any
    slice_mask: Any
    series_mask: Any
    plane: Any
    fluid: Any
    fat: Any
    geometry_features: Any
    geometry_mask: Any
    study_ids: Any

def assert_inference_fields(keys: object) -> None:
    names = set(keys)
    if names != set(INFERENCE_FIELDS):
        raise ValueError(f"Invalid inference keys: missing={set(INFERENCE_FIELDS) - names}, unexpected={names - set(INFERENCE_FIELDS)}")
    if names.intersection(FORBIDDEN_INFERENCE_FIELDS):
        raise ValueError('Report/label supervision has leaked into inference')
