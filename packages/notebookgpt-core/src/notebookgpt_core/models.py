"""Dados do caderno do NotebookGPT."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


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
