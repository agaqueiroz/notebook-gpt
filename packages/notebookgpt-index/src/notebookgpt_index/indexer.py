"""Orquestracao: o que indexar, o que reaproveitar e o que devolver na busca."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass

from notebookgpt_core import Chunk, Source, split_into_chunks
from notebookgpt_core.context import TOTAL_SOURCE_CHARS

from notebookgpt_index.store import DimensionMismatchError, VectorIndex

DEFAULT_CANDIDATES = 12
"""Quantos vizinhos pedir ao banco antes de cortar pelo orcamento.

Nao ha limiar de similaridade: medi `dot` = 0,81 entre dois textos sem nenhuma
relacao, porque o e5 comprime tudo numa faixa alta. Um "score minimo" seria chute.
O que funciona e ranking -- pedir alguns candidatos e cortar por quanto cabe.
"""


def content_digest(content: str) -> str:
    """Identidade do conteudo de uma fonte, para saber se precisa reindexar."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncodingStats:
    """O que a ultima sincronizacao mandou ao embedding, e como ficou o indice."""

    model: str
    dimensions: int | None
    rebuilt: bool
    sources_encoded: int
    chunks_encoded: int
    chars_encoded: int
    seconds: float
    indexed_sources: int
    indexed_chunks: int

    @property
    def chunks_per_second(self) -> float:
        return self.chunks_encoded / self.seconds if self.seconds > 0 else 0.0

    @property
    def chars_per_chunk(self) -> float:
        return self.chars_encoded / self.chunks_encoded if self.chunks_encoded else 0.0


class Indexer:
    """Mantem o banco vetorial em dia com as fontes e responde as buscas."""

    def __init__(self, store: VectorIndex, client, candidates: int = DEFAULT_CANDIDATES):
        self.store = store
        self.client = client
        self.candidates = candidates
        self.last_encoding: EncodingStats | None = None

    async def sync(self, sources: Sequence[Source]) -> int:
        """Poe o banco em dia com `sources`; devolve quantas fontes reindexou.

        Compara o sha256 do conteudo com o que esta gravado, entao reabrir o
        aplicativo e perguntar de novo nao gasta uma rodada de embeddings. E a
        diferenca entre um PoC usavel e um que espera a cada pergunta.

        Vetores de modelos diferentes nao se comparam, entao o indice e refeito
        do zero em dois casos: quando o modelo configurado nao e o que gerou o
        indice, e quando o servidor devolve vetores de outro tamanho sob o mesmo
        nome -- o serv04 ja trocou o e5 pelo qwen3-embedding assim.
        """
        model = self.client.settings.embedding_model
        rebuilt = self.store.embedding_model != model
        if rebuilt:
            self.store.reset(model)

        try:
            return await self._sync(sources, rebuilt=rebuilt)
        except DimensionMismatchError:
            self.store.reset(model)
            return await self._sync(sources, rebuilt=True)

    async def _sync(self, sources: Sequence[Source], *, rebuilt: bool) -> int:
        indexed = self.store.indexed()
        current = {source.name for source in sources}

        for gone in set(indexed) - current:
            self.store.drop_source(gone)

        reindexed = 0
        chunks_encoded = 0
        chars_encoded = 0
        seconds = 0.0
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

            # So o tempo do embedding: e ele que mede o servidor, a gravacao no
            # sqlite e irrelevante perto disso.
            started = time.perf_counter()
            vectors = await self.client.embed(
                [chunk.content for chunk in chunks],
                prefix=self.client.settings.passage_prefix,
            )
            seconds += time.perf_counter() - started
            self.store.replace_source(source.name, digest, chunks, vectors)
            reindexed += 1
            chunks_encoded += len(chunks)
            chars_encoded += sum(len(chunk.content) for chunk in chunks)

        self.last_encoding = EncodingStats(
            # O nome que o servidor diz ter usado, quando o cliente sabe: o pedido
            # pode dizer um modelo e a resposta vir de outro.
            model=getattr(self.client, "served_embedding_model", None) or self.client.settings.embedding_model,
            dimensions=self.store.dimensions,
            rebuilt=rebuilt,
            sources_encoded=reindexed,
            chunks_encoded=chunks_encoded,
            chars_encoded=chars_encoded,
            seconds=seconds,
            indexed_sources=len(self.store.indexed()),
            indexed_chunks=self.store.count(),
        )
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
