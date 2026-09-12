"""Banco vetorial embutido do NotebookGPT: indexa as fontes e recupera trechos."""

from notebookgpt_index.indexer import DEFAULT_CANDIDATES, Indexer
from notebookgpt_index.store import EMBEDDING_DIMS, VectorIndex, VectorIndexError

__all__ = [
    "DEFAULT_CANDIDATES",
    "EMBEDDING_DIMS",
    "Indexer",
    "VectorIndex",
    "VectorIndexError",
]
