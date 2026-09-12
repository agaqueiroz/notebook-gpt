"""Dominio do NotebookGPT: caderno, fontes TXT, persistencia e contexto."""

from notebookgpt_core.context import build_messages, format_sources
from notebookgpt_core.models import Message, NotebookData, Source
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
    "Message",
    "NotebookData",
    "Source",
    "StorageError",
    "UnsupportedSourceError",
    "build_messages",
    "format_sources",
    "is_supported",
    "load_notebook",
    "notebook_from_text",
    "notebook_to_text",
    "read_txt_source",
    "save_notebook",
    "source_from_bytes",
]
