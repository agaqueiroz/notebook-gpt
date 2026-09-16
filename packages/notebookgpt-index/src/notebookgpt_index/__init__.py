"""Banco vetorial embutido do NotebookGPT: indexa as fontes e recupera trechos."""

from notebookgpt_index.indexer import DEFAULT_CANDIDATES, EncodingStats, Indexer
from notebookgpt_index.store import EMBEDDING_DIMS, DimensionMismatchError, VectorIndex, VectorIndexError

__all__ = [
    "DEFAULT_CANDIDATES",
    "EMBEDDING_DIMS",
    "DimensionMismatchError",
    "EncodingStats",
    "Indexer",
    "VectorIndex",
    "VectorIndexError",
]
