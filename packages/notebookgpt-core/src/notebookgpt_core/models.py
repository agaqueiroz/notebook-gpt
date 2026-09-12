"""Dados do caderno do NotebookGPT."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Um trecho de uma fonte, na granularidade em que ela e indexada.

    `ordinal` e `total` existem para a resposta poder dizer de onde o trecho veio
    ("trecho 2 de 7"): num PoC de RAG, ver o que o modelo recebeu e metade do valor.
    """

    source_name: str
    ordinal: int
    total: int
    content: str


@dataclass
class Message:
    role: str
    content: str
    # Os trechos que embasaram esta resposta, quando ela veio do banco vetorial.
    # Ficam gravados junto: reabrir o caderno e poder conferir a procedencia de uma
    # resposta antiga vale mais que os bytes que isso custa.
    passages: list[Chunk] = field(default_factory=list)


@dataclass
class Source:
    """Um arquivo de contexto. Nesta etapa, sempre um TXT.

    O conteudo fica junto do nome porque o caderno e persistido como um unico
    arquivo: e o que faz as fontes sobreviverem a um reinicio tambem no modo web,
    onde nao existe diretorio de documentos para reler o original.
    """

    name: str
    content: str = ""


@dataclass
class NotebookData:
    sources: list[Source] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    notes: str = ""
