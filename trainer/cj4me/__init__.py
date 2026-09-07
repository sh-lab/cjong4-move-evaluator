"""Training support for cjong4-move-evaluator."""

from .dataset import (
    DATASET_FORMAT_VERSION,
    DATASET_HEADER_SIZE,
    DATASET_MAGIC,
    DATASET_RECORD_SIZE,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    CJ4MEDataset,
    DatasetHeader,
    DatasetV1,
    DatasetV2,
    DatasetV3,
    read_dataset,
)
from .model import MODEL_DIMENSIONS, MoveEvaluator, checkpoint_metadata
from .quantize import (
    QuantizedModel,
    infer_int8_reference,
    quantize_model,
    validate_quantized_model,
)
from .reward import (
    DEFAULT_PERSONALITY_WEIGHT,
    DEFAULT_REWARD_SCALE,
    PERSONALITIES,
    TeacherDataset,
    build_teacher_dataset,
    compose_targets,
    personality_component,
)

__all__ = [
    "CJ4MEDataset",
    "DATASET_FORMAT_VERSION",
    "DATASET_HEADER_SIZE",
    "DATASET_MAGIC",
    "DATASET_RECORD_SIZE",
    "DEFAULT_PERSONALITY_WEIGHT",
    "DEFAULT_REWARD_SCALE",
    "DatasetHeader",
    "DatasetV1",
    "DatasetV2",
    "DatasetV3",
    "FEATURE_COUNT",
    "FEATURE_SCHEMA_VERSION",
    "MODEL_DIMENSIONS",
    "MoveEvaluator",
    "PERSONALITIES",
    "QuantizedModel",
    "TeacherDataset",
    "build_teacher_dataset",
    "checkpoint_metadata",
    "compose_targets",
    "infer_int8_reference",
    "quantize_model",
    "personality_component",
    "read_dataset",
    "validate_quantized_model",
]
