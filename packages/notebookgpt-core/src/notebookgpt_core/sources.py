"""Leitura dos arquivos de contexto. Nesta etapa, somente TXT."""

from __future__ import annotations

from pathlib import Path

from notebookgpt_core.models import Source

SUPPORTED_SUFFIXES = frozenset({".txt"})


class UnsupportedSourceError(Exception):
    """O arquivo escolhido nao e um formato aceito como fonte."""


def is_supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_SUFFIXES


def read_txt_source(path: Path | str, name: str | None = None) -> Source:
    """Le um TXT como fonte de contexto.

    `errors="replace"` em vez de falhar: um TXT gravado em outra codificacao ainda
    e util como contexto, e perder um acento e melhor que perder o arquivo.
    """
    path = Path(path)
    if not is_supported(path.name):
        raise UnsupportedSourceError(
            f"'{path.name}' nao e um formato aceito. Nesta versao apenas arquivos .txt podem ser fontes."
        )

    content = path.read_text(encoding="utf-8", errors="replace")
    return Source(name=name or path.name, content=content)


def source_from_bytes(name: str, data: bytes) -> Source:
    """Le uma fonte a partir do conteudo bruto, sem passar pelo disco.

    E o caminho que o seletor de arquivos do Flet usa: no modo web ele nunca
    devolve um caminho, so os bytes.
    """
    if not is_supported(name):
        raise UnsupportedSourceError(
            f"'{name}' nao e um formato aceito. Nesta versao apenas arquivos .txt podem ser fontes."
        )

    return Source(name=name, content=data.decode("utf-8", errors="replace"))
