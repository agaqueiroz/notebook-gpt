"""Quebra das fontes TXT em trechos indexaveis.

Os tamanhos vem de medicao contra o multilingual-e5-small do serv04, nao de
intuicao: o modelo trunca em 511 tokens em silencio, e em portugues medi ~3,9
caracteres por token (312 chars -> 83 tokens; 1248 -> 317). Ou seja, o teto real
fica perto de 2000 caracteres. CHUNK_CHARS = 1200 da ~310 tokens, com folga para
o prefixo que o servidor (ou o cliente) acrescenta e para palavras longas.
"""

from __future__ import annotations

import re

from notebookgpt_core.models import Chunk, Source

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

# Fronteiras preferidas, da melhor para a pior: paragrafo, fim de frase, quebra de
# linha, espaco. Um corte no meio de uma palavra e o ultimo recurso.
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?…][\s\"')\]]*")


def _best_split(text: str, limit: int) -> int:
    """Onde cortar `text` para o trecho nao passar de `limit` caracteres.

    Procura a fronteira mais tardia dentro do limite, para os trechos saírem
    proximos do tamanho cheio em vez de picados.
    """
    if len(text) <= limit:
        return len(text)

    window = text[:limit]
    # A busca comeca depois da metade: uma fronteira muito no inicio produziria um
    # trecho curto demais e multiplicaria as chamadas de embedding.
    floor = limit // 2

    paragraphs = [match.end() for match in _PARAGRAPH.finditer(window) if match.end() > floor]
    if paragraphs:
        return paragraphs[-1]

    sentences = [match.end() for match in _SENTENCE_END.finditer(window) if match.end() > floor]
    if sentences:
        return sentences[-1]

    newline = window.rfind("\n", floor)
    if newline != -1:
        return newline + 1

    space = window.rfind(" ", floor)
    if space != -1:
        return space + 1

    return limit


def split_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Os pedacos de `text`, com sobreposicao entre vizinhos.

    A sobreposicao existe para uma frase partida na fronteira aparecer inteira em
    pelo menos um dos lados -- e o que evita que a resposta se perca exatamente na
    costura entre dois trechos.
    """
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= size:
        return [clean]

    pieces: list[str] = []
    start = 0
    while start < len(clean):
        remaining = clean[start:]
        cut = _best_split(remaining, size)
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        if start + cut >= len(clean):
            break
        # Recua a sobreposicao, mas sempre avanca: sem o max(), um corte menor que a
        # sobreposicao faria o laco andar para tras e nunca terminar.
        start += max(cut - overlap, 1)

    return pieces


def split_into_chunks(source: Source, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Uma fonte inteira como trechos numerados."""
    pieces = split_text(source.content, size=size, overlap=overlap)
    total = len(pieces)
    return [
        Chunk(source_name=source.name, ordinal=ordinal, total=total, content=piece)
        for ordinal, piece in enumerate(pieces, start=1)
    ]
