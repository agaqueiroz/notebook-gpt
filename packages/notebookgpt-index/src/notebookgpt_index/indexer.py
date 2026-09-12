"""Orquestracao: o que indexar, o que reaproveitar e o que devolver na busca."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from notebookgpt_core import Chunk, Source, split_into_chunks
from notebookgpt_core.context import TOTAL_SOURCE_CHARS

from notebookgpt_index.store import VectorIndex

DEFAULT_CANDIDATES = 12
"""Quantos vizinhos pedir ao banco antes de cortar pelo orcamento.

Nao ha limiar de similaridade: medi `dot` = 0,81 entre dois textos sem nenhuma
relacao, porque o e5 comprime tudo numa faixa alta. Um "score minimo" seria chute.
O que funciona e ranking -- pedir alguns candidatos e cortar por quanto cabe.
"""


def content_digest(content: str) -> str:
    """Identidade do conteudo de uma fonte, para saber se precisa reindexar."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Indexer:
    """Mantem o banco vetorial em dia com as fontes e responde as buscas."""

    def __init__(self, store: VectorIndex, client, candidates: int = DEFAULT_CANDIDATES):
        self.store = store
        self.client = client
        self.candidates = candidates

    async def sync(self, sources: Sequence[Source]) -> int:
        """Poe o banco em dia com `sources`; devolve quantas fontes reindexou.

        Compara o sha256 do conteudo com o que esta gravado, entao reabrir o
        aplicativo e perguntar de novo nao gasta uma rodada de embeddings. E a
        diferenca entre um PoC usavel e um que espera a cada pergunta.
        """
        indexed = self.store.indexed()
        current = {source.name for source in sources}

        for gone in set(indexed) - current:
            self.store.drop_source(gone)

        reindexed = 0
        for source in sources:
            digest = content_digest(source.content)
            if indexed.get(source.name) == digest:
                continue

            chunks = split_into_chunks(source)
            if not chunks:
                # Fonte vazia: nada a indexar, mas o digest fica gravado para nao
                # tentar de novo a cada pergunta.
                self.store.replace_source(source.name, digest, [], [])
                reindexed += 1
                continue

            vectors = await self.client.embed(
                [chunk.content for chunk in chunks],
                prefix=self.client.settings.passage_prefix,
            )
            self.store.replace_source(source.name, digest, chunks, vectors)
            reindexed += 1

        return reindexed

    async def retrieve(self, prompt: str, budget: int = TOTAL_SOURCE_CHARS) -> list[Chunk]:
        """Os trechos mais proximos da pergunta que couberem no orcamento.

        Devolvidos na ordem do ranking: o mais proximo primeiro, que e tambem a
        ordem em que entram no contexto.
        """
        if self.store.count() == 0:
            return []

        query = await self.client.embed([prompt], prefix=self.client.settings.query_prefix)
        found = self.store.search(query[0], self.candidates)

        chosen: list[Chunk] = []
        remaining = budget
        for chunk, _distance in found:
            if len(chunk.content) > remaining:
                continue
            chosen.append(chunk)
            remaining -= len(chunk.content)

        # Se nem o melhor trecho cabe, mande o melhor cortado -- voltar de mao vazia
        # seria pior: o modelo perderia o unico contexto relevante que existe.
        if not chosen and found:
            best = found[0][0]
            chosen = [Chunk(best.source_name, best.ordinal, best.total, best.content[:budget])]

        return chosen
