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


def _block(name: str, body: str) -> str:
    """Uma fonte como texto delimitado, sem cabecalhos Markdown.

    O formato importa mais do que parece. Medido contra o llama-3.2-1b do
    serv04, com temperature=0, em cinco perguntas cuja resposta esta na fonte:
    cabecalhos Markdown (`## Fontes anexadas` / `### arquivo.txt`) acertam 2/5 e
    chegam a responder "nao posso responder a esta pergunta"; o texto delimitado
    abaixo acerta 4/5. Um modelo pequeno trata o cabecalho como conteudo a
    interpretar, nao como moldura.
    """
    return f'Arquivo "{name}":\n"""\n{body}\n"""'


def format_sources(sources: list[Source]) -> str:
    """Concatena as fontes em um bloco unico, respeitando o orcamento de caracteres."""
    if not sources:
        return NO_SOURCES_NOTE

    blocks: list[str] = []
    remaining = TOTAL_SOURCE_CHARS
    for source in sources:
        if remaining <= 0:
            blocks.append(_block(source.name, TRUNCATION_MARK.strip()))
            continue
        body = _truncate(source.content, min(PER_SOURCE_CHARS, remaining))
        remaining -= len(body)
        blocks.append(_block(source.name, body))

    return "\n\n".join(blocks)


def build_messages(notebook: NotebookData, prompt: str) -> list[Message]:
    """Monta system + historico recente + (fontes junto da pergunta).

    As fontes vao no turno do usuario, nao em `system`. Medido contra o
    llama-3.2-1b do serv04: com as fontes em `system` -- em uma mensagem ou em
    duas -- o modelo responde "nao tenho acesso a essa informacao" mesmo com o
    texto ali; no turno do usuario ele passa a usar o conteudo. So o turno atual
    carrega as fontes, para nao repeti-las em cada mensagem do historico.
    """
    history = notebook.messages[-HISTORY_TURNS:] if HISTORY_TURNS else []
    question = f"{format_sources(notebook.sources)}\n\nPergunta: {prompt}"
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        *[Message(role=message.role, content=message.content) for message in history],
        Message(role="user", content=question),
    ]
