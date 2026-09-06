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
    read_dataset,
)
from .model import MODEL_DIMENSIONS, MoveEvaluator, checkpoint_metadata
from .quantize import (
    QuantizedModel,
    infer_int8_reference,
    quantize_model,
    validate_quantized_model,
)

__all__ = [
    "CJ4MEDataset",
    "DATASET_FORMAT_VERSION",
    "DATASET_HEADER_SIZE",
    "DATASET_MAGIC",
    "DATASET_RECORD_SIZE",
    "DatasetHeader",
    "DatasetV1",
    "DatasetV2",
    "FEATURE_COUNT",
    "FEATURE_SCHEMA_VERSION",
    "MODEL_DIMENSIONS",
    "MoveEvaluator",
    "QuantizedModel",
    "checkpoint_metadata",
    "infer_int8_reference",
    "quantize_model",
    "read_dataset",
    "validate_quantized_model",
]
