"""Montagem da conversa enviada ao servidor de inferencia."""

from __future__ import annotations

from notebookgpt_core.models import Chunk, Message, NotebookData, Source

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


def _block(name: str, body: str, *, label: str = "") -> str:
    """Uma fonte como texto delimitado, sem cabecalhos Markdown.

    O formato importa mais do que parece. Medido contra o llama-3.2-1b do serv04,
    com temperature=0, em cinco perguntas cuja resposta esta na fonte anexada:

        cabecalhos Markdown em `system`      1/5
        cabecalhos Markdown no turno do user 0/5
        texto delimitado em `system`         0/5
        texto delimitado no turno do user    4/5  <- este

    Nenhuma das duas escolhas (formato e posicao) resolve sozinha. Um modelo
    pequeno ignora o que esta em `system` e trata o cabecalho Markdown como
    conteudo a interpretar, nao como moldura. Ver build_messages para a posicao.
    """
    return f'Arquivo "{name}"{label}:\n"""\n{body}\n"""'


def format_passages(passages: list[Chunk]) -> str:
    """Os trechos recuperados, no mesmo formato delimitado das fontes inteiras.

    O rotulo ganha a procedencia ("trecho 2 de 7") porque a resposta cita o arquivo
    e o usuario precisa poder conferir de onde o modelo tirou o que disse.
    """
    if not passages:
        return NO_SOURCES_NOTE

    return "\n\n".join(
        _block(
            passage.source_name,
            passage.content,
            label=f" (trecho {passage.ordinal} de {passage.total})",
        )
        for passage in passages
    )


def fits_whole(sources: list[Source]) -> bool:
    """As fontes cabem inteiras no orcamento, sem precisar de recuperacao?

    Recuperar num corpus pequeno so subtrai: os trechos descartados teriam cabido
    de qualquer jeito. Acima do orcamento e que a busca por similaridade ganha.
    """
    return sum(len(source.content) for source in sources) <= TOTAL_SOURCE_CHARS


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


def build_messages(
    notebook: NotebookData,
    prompt: str,
    passages: list[Chunk] | None = None,
) -> list[Message]:
    """Monta system + historico recente + (contexto junto da pergunta).

    O contexto vai no turno do usuario, nao em `system`. Medido contra o
    llama-3.2-1b do serv04: com as fontes em `system` -- em uma mensagem ou em
    duas -- o modelo responde "nao tenho acesso a essa informacao" mesmo com o
    texto ali; no turno do usuario ele passa a usar o conteudo. So o turno atual
    carrega o contexto, para nao repeti-lo em cada mensagem do historico.

    `passages` sao os trechos vindos do banco vetorial. Sem eles, as fontes entram
    inteiras (truncadas pelo orcamento) -- o caminho de quando tudo cabe.
    """
    history = notebook.messages[-HISTORY_TURNS:] if HISTORY_TURNS else []
    context = format_passages(passages) if passages is not None else format_sources(notebook.sources)
    question = f"{context}\n\nPergunta: {prompt}"
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        *[Message(role=message.role, content=message.content) for message in history],
        Message(role="user", content=question),
    ]
