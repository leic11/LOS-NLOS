# models/STCA/__init__.py
"""
STCA Model Subpackage
=====================

This subpackage contains all modules related to the Spatiotemporal Cross-Attention (STCA) model:

- stca_model: Main STCA model class
- spatial_encoder: AAM (Attention Aggregation Module) for spatial feature extraction
- temporal_encoder: LSTM-TFE for temporal feature extraction
- cross_attention: Cross-attention module for space-time fusion
- sparse_representation: Sparse representation module with L1 regularization

Usage:
    from models.STCA import STCAModel
    from models.STCA.spatial_encoder import SpatialEncoder
    from models.STCA.temporal_encoder import TemporalEncoder
    from models.STCA.cross_attention import CrossAttention
    from models.STCA.sparse_representation import SparseRepresentation
"""

from .stca_model import STCAModel
from .spatial_encoder import SpatialEncoder
from .temporal_encoder import TemporalEncoder
from .cross_attention import CrossAttention
from .sparse_representation import SparseRepresentation

__all__ = [
    'STCAModel',
    'SpatialEncoder',
    'TemporalEncoder',
    'CrossAttention',
    'SparseRepresentation',
    'SparseLoss',
]
