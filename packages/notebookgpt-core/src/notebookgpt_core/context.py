"""Montagem da conversa enviada ao servidor de inferencia."""

from __future__ import annotations

from notebookgpt_core.models import Message, NotebookData, Source

# O servidor local roda com LIS_MAX_MODEL_LENGTH=4096, entao o contexto tem
# orcamento: um TXT inteiro nao cabe e mandar tudo faz o servidor recusar o pedido.
PER_SOURCE_CHARS = 3000
TOTAL_SOURCE_CHARS = 6000
HISTORY_TURNS = 6
TRUNCATION_MARK = "\n… [fonte truncada]"

SYSTEM_PROMPT = (
    "Voce e o NotebookGPT, um assistente que responde com base nas fontes anexadas pelo usuario. "
    "Responda sempre em portugues do Brasil e formate a resposta em Markdown. "
    "Use apenas o que estiver nas fontes; se elas nao contiverem a resposta, diga isso claramente "
    "em vez de inventar. Quando usar uma fonte, mencione o nome do arquivo."
)

NO_SOURCES_NOTE = "O usuario ainda nao anexou nenhuma fonte. Responda com o que souber e avise que nao ha fontes."


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + TRUNCATION_MARK


def format_sources(sources: list[Source]) -> str:
    """Concatena as fontes em um bloco unico, respeitando o orcamento de caracteres."""
    if not sources:
        return NO_SOURCES_NOTE

    blocks: list[str] = []
    remaining = TOTAL_SOURCE_CHARS
    for source in sources:
        if remaining <= 0:
            blocks.append(f"### {source.name}\n{TRUNCATION_MARK.strip()}")
            continue
        body = _truncate(source.content, min(PER_SOURCE_CHARS, remaining))
        remaining -= len(body)
        blocks.append(f"### {source.name}\n{body}")

    return "## Fontes anexadas\n\n" + "\n\n".join(blocks)


def build_messages(notebook: NotebookData, prompt: str) -> list[Message]:
    """Monta system + fontes + historico recente + pergunta atual."""
    history = notebook.messages[-HISTORY_TURNS:] if HISTORY_TURNS else []
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="system", content=format_sources(notebook.sources)),
        *[Message(role=message.role, content=message.content) for message in history],
        Message(role="user", content=prompt),
    ]
