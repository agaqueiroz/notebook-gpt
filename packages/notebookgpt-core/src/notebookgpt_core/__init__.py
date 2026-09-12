"""Dominio do NotebookGPT: caderno, fontes TXT, persistencia e contexto."""

from notebookgpt_core.chunking import split_into_chunks, split_text
from notebookgpt_core.context import build_messages, fits_whole, format_passages, format_sources
from notebookgpt_core.models import Chunk, Message, NotebookData, Source
from notebookgpt_core.sources import (
    UnsupportedSourceError,
    is_supported,
    read_txt_source,
    source_from_bytes,
)
from notebookgpt_core.storage import (
    StorageError,
    load_notebook,
    notebook_from_text,
    notebook_to_text,
    save_notebook,
)

__all__ = [
    "Chunk",
    "Message",
    "NotebookData",
    "Source",
    "StorageError",
    "UnsupportedSourceError",
    "build_messages",
    "fits_whole",
    "format_passages",
    "format_sources",
    "is_supported",
    "load_notebook",
    "notebook_from_text",
    "notebook_to_text",
    "read_txt_source",
    "save_notebook",
    "source_from_bytes",
    "split_into_chunks",
    "split_text",
]
