"""A orquestracao da indexacao, com um embedder dublê que conta o que foi pedido."""

import math
import random

import pytest
from notebookgpt_core import Source
from notebookgpt_index import EMBEDDING_DIMS, DimensionMismatchError, Indexer, VectorIndex
from notebookgpt_inference import InferenceSettings

GRANDE = "Frase de tamanho razoavel sobre previdencia social brasileira. " * 60


class FakeEmbedder:
    """Substitui o LocalInferenceClient: mesma superficie, sem rede.

    Gera um vetor unitario deterministico por texto, entao o mesmo texto sempre
    embute igual -- o que permite testar busca sem depender do servidor.
    """

    def __init__(self, settings: InferenceSettings | None = None, max_inputs: int = 64):
        self.settings = settings or InferenceSettings()
        self.max_inputs = max_inputs
        self.dimensions = EMBEDDING_DIMS
        self.batches: list[list[str]] = []

    @property
    def calls(self) -> int:
        return len(self.batches)

    @property
    def embedded(self) -> list[str]:
        return [text for batch in self.batches for text in batch]

    async def embed(self, texts, *, prefix=""):
        # Divide em lotes como o cliente de verdade: o limite de 64 do servidor e
        # responsabilidade de LocalInferenceClient.embed, nao do Indexer. Ver
        # tests/inference/test_client.py, que fixa a divisao no lugar onde ela mora.
        prefixed = [f"{prefix}{text}" for text in texts]
        vectors = []
        for start in range(0, len(prefixed), self.max_inputs):
            batch = prefixed[start : start + self.max_inputs]
            self.batches.append(batch)
            for text in batch:
                rng = random.Random(text)
                values = [rng.gauss(0, 1) for _ in range(self.dimensions)]
                norm = math.sqrt(sum(value * value for value in values))
                vectors.append([value / norm for value in values])
        return vectors


@pytest.fixture
def indexer():
    with VectorIndex(None) as store:
        yield Indexer(store, FakeEmbedder())


async def test_syncing_a_source_indexes_its_chunks(indexer):
    reindexadas = await indexer.sync([Source(name="a.txt", content=GRANDE)])

    assert reindexadas == 1
    assert indexer.store.count() > 1
    assert set(indexer.store.indexed()) == {"a.txt"}


async def test_an_unchanged_source_is_not_embedded_again(indexer):
    fontes = [Source(name="a.txt", content=GRANDE)]
    await indexer.sync(fontes)
    depois_da_primeira = indexer.client.calls

    reindexadas = await indexer.sync(fontes)

    assert reindexadas == 0
    assert indexer.client.calls == depois_da_primeira, "reembutiu uma fonte que nao mudou"


async def test_changed_content_is_reindexed(indexer):
    await indexer.sync([Source(name="a.txt", content=GRANDE)])
    antes = indexer.store.count()

    reindexadas = await indexer.sync([Source(name="a.txt", content=GRANDE + " Um paragrafo novo no fim.")])

    assert reindexadas == 1
    assert indexer.store.count() >= antes
    assert len(indexer.store.indexed()) == 1


async def test_a_removed_source_leaves_the_index(indexer):
    await indexer.sync([Source(name="a.txt", content=GRANDE), Source(name="b.txt", content=GRANDE * 2)])

    await indexer.sync([Source(name="b.txt", content=GRANDE * 2)])

    assert set(indexer.store.indexed()) == {"b.txt"}
    assert all(chunk.source_name == "b.txt" for chunk, _ in indexer.store.search([0.0] * EMBEDDING_DIMS, 5))


async def test_indexing_respects_the_server_batch_limit(indexer):
    # Texto grande o suficiente para passar de 64 trechos e forcar mais de um lote.
    enorme = "Paragrafo distinto numero {}. ".format
    conteudo = "\n\n".join(enorme(i) * 30 for i in range(120))

    await indexer.sync([Source(name="enorme.txt", content=conteudo)])

    assert indexer.client.calls > 1, "deveria ter quebrado em mais de um lote"
    assert all(len(batch) <= 64 for batch in indexer.client.batches)


async def test_passages_are_embedded_with_the_passage_prefix(indexer):
    await indexer.sync([Source(name="a.txt", content=GRANDE)])

    assert all(text.startswith("passage: ") for text in indexer.client.embedded)


async def test_the_question_is_embedded_with_the_query_prefix(indexer):
    await indexer.sync([Source(name="a.txt", content=GRANDE)])
    indexer.client.batches.clear()

    await indexer.retrieve("qual e a regra?")

    assert indexer.client.embedded == ["query: qual e a regra?"]


async def test_retrieval_on_an_empty_index_returns_nothing_and_asks_nobody(indexer):
    assert await indexer.retrieve("qualquer coisa") == []
    assert indexer.client.calls == 0, "nao faz sentido embutir a pergunta sem indice"


async def test_retrieval_finds_the_chunk_that_matches_the_question(indexer):
    # O dublê embute por texto, entao uma pergunta identica a um trecho recupera
    # exatamente aquele trecho -- o suficiente para provar a ligacao busca/indice.
    fonte = Source(name="fatos.txt", content="\n\n".join(f"O fato numero {i} e unico e distinto." for i in range(20)))
    await indexer.sync([fonte])
    trechos = indexer.store.search([0.0] * EMBEDDING_DIMS, 1)
    alvo = trechos[0][0].content

    recuperados = await indexer.retrieve(alvo)

    assert recuperados
    assert any(trecho.content == alvo for trecho in recuperados)


async def test_retrieval_stops_at_the_budget(indexer):
    await indexer.sync([Source(name="a.txt", content=GRANDE * 4)])

    recuperados = await indexer.retrieve("qualquer coisa", budget=1500)

    assert recuperados
    assert sum(len(trecho.content) for trecho in recuperados) <= 1500


async def test_a_budget_smaller_than_any_chunk_still_returns_something(indexer):
    # Voltar de mao vazia seria pior: o modelo perderia o unico contexto relevante.
    await indexer.sync([Source(name="a.txt", content=GRANDE)])

    recuperados = await indexer.retrieve("qualquer coisa", budget=50)

    assert len(recuperados) == 1
    assert len(recuperados[0].content) <= 50


async def test_sync_records_the_encoding_metrics(indexer):
    assert indexer.last_encoding is None

    await indexer.sync([Source(name="a.txt", content=GRANDE)])
    metricas = indexer.last_encoding

    assert metricas.model == indexer.client.settings.embedding_model
    assert metricas.dimensions == EMBEDDING_DIMS
    assert metricas.sources_encoded == 1
    assert metricas.chunks_encoded == len(indexer.client.embedded) == indexer.store.count()
    assert metricas.chars_encoded > 0
    assert metricas.indexed_sources == 1
    assert metricas.indexed_chunks == indexer.store.count()


async def test_a_sync_with_nothing_new_reports_zero_encoded_but_keeps_the_index_totals(indexer):
    fontes = [Source(name="a.txt", content=GRANDE)]
    await indexer.sync(fontes)

    await indexer.sync(fontes)
    metricas = indexer.last_encoding

    assert (metricas.sources_encoded, metricas.chunks_encoded, metricas.seconds) == (0, 0, 0.0)
    assert metricas.chunks_per_second == 0.0
    assert metricas.indexed_chunks == indexer.store.count() > 0


async def test_a_server_that_changes_the_vector_size_rebuilds_the_index(indexer):
    # O caso real: pedindo o e5 (384), o serv04 passou a responder com o
    # qwen3-embedding (1024) sob o mesmo nome. Antes isso derrubava o aplicativo.
    fontes = [Source(name="a.txt", content=GRANDE), Source(name="b.txt", content=GRANDE * 2)]
    await indexer.sync(fontes)
    assert indexer.store.dimensions == EMBEDDING_DIMS

    indexer.client.dimensions = 1024
    await indexer.sync([*fontes, Source(name="c.txt", content=GRANDE * 3)])

    assert indexer.store.dimensions == 1024
    assert set(indexer.store.indexed()) == {"a.txt", "b.txt", "c.txt"}
    assert indexer.last_encoding.rebuilt
    assert indexer.last_encoding.sources_encoded == 3
    assert await indexer.retrieve("qualquer coisa")


async def test_changing_the_configured_model_rebuilds_the_index(indexer):
    fontes = [Source(name="a.txt", content=GRANDE)]
    await indexer.sync(fontes)
    antes = indexer.client.calls

    indexer.client.settings = InferenceSettings(embedding_model="outro-modelo")
    reindexadas = await indexer.sync(fontes)

    assert reindexadas == 1
    assert indexer.client.calls > antes
    assert indexer.store.embedding_model == "outro-modelo"


async def test_a_question_embedded_by_another_model_is_refused_clearly(indexer):
    await indexer.sync([Source(name="a.txt", content=GRANDE)])
    indexer.client.dimensions = 1024

    with pytest.raises(DimensionMismatchError):
        await indexer.retrieve("qualquer coisa")


async def test_an_empty_source_is_recorded_and_not_retried(indexer):
    vazia = [Source(name="vazio.txt", content="   ")]
    await indexer.sync(vazia)
    depois = indexer.client.calls

    assert await indexer.sync(vazia) == 0
    assert indexer.client.calls == depois
