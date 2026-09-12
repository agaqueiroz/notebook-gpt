"""Persistencia do caderno em um TXT legivel e portatil (conteudo JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from notebookgpt_core.models import Chunk, Message, NotebookData, Source


class StorageError(Exception):
    """Nao foi possivel gravar o caderno."""


def _source_from_raw(raw: object) -> Source | None:
    """Le uma fonte nos dois formatos ja gravados por versoes do aplicativo.

    Cadernos antigos guardavam `sources` como uma lista de nomes; os atuais
    guardam nome e conteudo. O formato antigo carrega sem conteudo -- a fonte
    aparece na lateral e o usuario pode adiciona-la de novo para ter o texto.
    """
    if isinstance(raw, str):
        return Source(name=raw)
    if isinstance(raw, dict) and isinstance(raw.get("name"), str):
        return Source(name=raw["name"], content=str(raw.get("content", "")))
    return None


def _message_from_raw(raw: dict) -> Message:
    """Le uma mensagem, com ou sem os trechos que a embasaram.

    Mensagens gravadas antes do banco vetorial nao tem `passages`; elas continuam
    carregando, com a lista vazia.
    """
    passages = [
        Chunk(
            source_name=str(item["source_name"]),
            ordinal=int(item["ordinal"]),
            total=int(item["total"]),
            content=str(item["content"]),
        )
        for item in raw.get("passages", [])
        if isinstance(item, dict) and "source_name" in item
    ]
    return Message(role=str(raw["role"]), content=str(raw["content"]), passages=passages)


def notebook_from_text(text: str | None) -> NotebookData:
    """Converte o conteudo textual persistido em dados do caderno."""
    if not text:
        return NotebookData()

    try:
        raw = json.loads(text)
        sources = [source for source in map(_source_from_raw, raw.get("sources", [])) if source is not None]
        return NotebookData(
            sources=sources,
            messages=[_message_from_raw(item) for item in raw.get("messages", [])],
            notes=str(raw.get("notes", "")),
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        # ValueError cobre json.JSONDecodeError e um `total` que nao vira inteiro;
        # KeyError, uma mensagem sem `role`. Caderno ilegivel abre vazio em vez de
        # derrubar o aplicativo -- o mesmo principio da falha de gravacao.
        return NotebookData()


def notebook_to_text(notebook: NotebookData) -> str:
    """Serializa o caderno para um TXT legivel, com conteudo JSON."""
    payload = {
        "sources": [asdict(source) for source in notebook.sources],
        "messages": [asdict(message) for message in notebook.messages],
        "notes": notebook.notes,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_notebook(path: Path) -> NotebookData:
    """Carrega o caderno de um arquivo TXT, se ele existir."""
    try:
        return notebook_from_text(path.read_text(encoding="utf-8"))
    except OSError:
        return NotebookData()


def save_notebook(path: Path, notebook: NotebookData) -> None:
    """Persiste o estado em um TXT legivel e portatil (conteudo JSON).

    Levanta StorageError -- e nao OSError cru -- para que a interface tenha uma
    mensagem pronta e possa seguir a conversa em vez de cair.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(notebook_to_text(notebook), encoding="utf-8")
    except OSError as error:
        # A mensagem cita o Windows de proposito: quando o Acesso Controlado a
        # Pastas bloqueia a escrita, o erro chega como ENOENT ("arquivo nao
        # encontrado") em vez de permissao negada, o que manda quem le para o
        # lado errado. Foi exatamente o que aconteceu com Documents dentro do
        # OneDrive.
        raise StorageError(
            f"Nao foi possivel gravar o caderno em {path}: {error}. "
            "No Windows, a Protecao contra Ransomware (Acesso Controlado a Pastas) "
            "pode estar bloqueando a escrita nessa pasta."
        ) from error
