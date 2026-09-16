"""O banco vetorial, sem tocar a rede.

Vetores unitarios sinteticos bastam: o que se testa aqui e o KNN e o ciclo de vida
das fontes, nao a qualidade dos embeddings.
"""

import math
import random

import pytest
from notebookgpt_core import Chunk
from notebookgpt_index import EMBEDDING_DIMS, DimensionMismatchError, VectorIndex


def unit(seed: int, dims: int = EMBEDDING_DIMS) -> list[float]:
    """Um vetor unitario reproduzivel -- como os que o servidor devolve."""
    rng = random.Random(seed)
    values = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def chunks_of(name: str, quantidade: int) -> list[Chunk]:
    return [
        Chunk(source_name=name, ordinal=i + 1, total=quantidade, content=f"trecho {i} de {name}")
        for i in range(quantidade)
    ]


@pytest.fixture
def index():
    with VectorIndex(None) as store:
        yield store


def test_a_fresh_index_is_empty(index):
    assert index.count() == 0
    assert index.indexed() == {}


def test_the_nearest_neighbour_of_a_vector_is_itself(index):
    chunks = chunks_of("a.txt", 8)
    vectors = [unit(i) for i in range(8)]
    index.replace_source("a.txt", "sha-1", chunks, vectors)

    for alvo in (0, 3, 7):
        encontrados = index.search(vectors[alvo], 3)
        assert encontrados[0][0].ordinal == chunks[alvo].ordinal
        assert encontrados[0][1] == pytest.approx(0.0, abs=1e-5)


def test_the_results_come_back_ordered_by_distance(index):
    index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 10), [unit(i) for i in range(10)])

    distancias = [distancia for _chunk, distancia in index.search(unit(99), 10)]

    assert distancias == sorted(distancias)


def test_the_chunk_text_comes_back_from_the_auxiliary_column(index):
    chunks = chunks_of("previdencia.txt", 3)
    vectors = [unit(i) for i in range(3)]
    index.replace_source("previdencia.txt", "sha-1", chunks, vectors)

    achado, _ = index.search(vectors[1], 1)[0]

    assert achado.content == chunks[1].content
    assert achado.source_name == "previdencia.txt"
    assert (achado.ordinal, achado.total) == (2, 3)


def test_replacing_a_source_does_not_duplicate_its_chunks(index):
    index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 5), [unit(i) for i in range(5)])
    index.replace_source("a.txt", "sha-2", chunks_of("a.txt", 2), [unit(i) for i in range(2)])

    assert index.count() == 2
    assert index.indexed() == {"a.txt": "sha-2"}


def test_dropping_a_source_leaves_the_others_alone(index):
    index.replace_source("a.txt", "sha-a", chunks_of("a.txt", 3), [unit(i) for i in range(3)])
    index.replace_source("b.txt", "sha-b", chunks_of("b.txt", 4), [unit(10 + i) for i in range(4)])

    index.drop_source("a.txt")

    assert index.count() == 4
    assert index.indexed() == {"b.txt": "sha-b"}
    assert all(chunk.source_name == "b.txt" for chunk, _ in index.search(unit(0), 4))


def test_an_empty_source_is_recorded_without_chunks(index):
    index.replace_source("vazio.txt", "sha-vazio", [], [])

    assert index.count() == 0
    assert index.indexed() == {"vazio.txt": "sha-vazio"}


def test_a_mismatch_between_chunks_and_vectors_is_refused(index):
    # `strict=True` no zip: associar o vetor de um trecho ao texto de outro seria um
    # bug silencioso e praticamente impossivel de notar depois.
    with pytest.raises(ValueError):
        index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 3), [unit(0), unit(1)])


def test_asking_for_no_neighbours_returns_nothing(index):
    index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 3), [unit(i) for i in range(3)])

    assert index.search(unit(0), 0) == []


def test_the_table_takes_the_size_of_the_first_vectors(index):
    assert index.dimensions is None

    index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 2), [unit(0, 1024), unit(1, 1024)])

    assert index.dimensions == 1024
    assert index.search(unit(0, 1024), 1)[0][0].ordinal == 1


def test_a_vector_of_another_size_is_refused_without_touching_the_index(index):
    index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 3), [unit(i) for i in range(3)])

    with pytest.raises(DimensionMismatchError):
        index.replace_source("a.txt", "sha-2", chunks_of("a.txt", 1), [unit(0, 1024)])
    with pytest.raises(DimensionMismatchError):
        index.search(unit(0, 1024), 1)

    assert index.indexed() == {"a.txt": "sha-1"}
    assert index.count() == 3


def test_reset_empties_the_index_and_accepts_a_new_size(index):
    index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 3), [unit(i) for i in range(3)])

    index.reset("qwen3-embedding-0.6b")
    index.replace_source("a.txt", "sha-1", chunks_of("a.txt", 2), [unit(0, 1024), unit(1, 1024)])

    assert (index.embedding_model, index.dimensions, index.count()) == ("qwen3-embedding-0.6b", 1024, 2)


def test_an_index_from_before_the_metadata_keeps_its_size(tmp_path):
    # Um `index.db` antigo: tabela de 384 dimensoes criada a mao, sem `meta`.
    import sqlite3

    import sqlite_vec

    path = tmp_path / "index.db"
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute(f"CREATE VIRTUAL TABLE chunks USING vec0(embedding float[{EMBEDDING_DIMS}], source_name text)")
    db.close()

    with VectorIndex(path) as antigo:
        assert antigo.dimensions == EMBEDDING_DIMS
        assert antigo.embedding_model is None


def test_the_index_survives_being_reopened(tmp_path):
    path = tmp_path / "sub" / "index.db"
    vectors = [unit(i) for i in range(4)]
    with VectorIndex(path) as store:
        store.replace_source("a.txt", "sha-1", chunks_of("a.txt", 4), vectors)

    with VectorIndex(path) as reaberto:
        assert reaberto.indexed() == {"a.txt": "sha-1"}
        assert reaberto.count() == 4
        assert reaberto.search(vectors[2], 1)[0][0].ordinal == 3
