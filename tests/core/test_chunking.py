from itertools import pairwise

from notebookgpt_core.chunking import CHUNK_CHARS, CHUNK_OVERLAP, split_into_chunks, split_text
from notebookgpt_core.models import Source

PARAGRAFO = "Esta e uma frase de tamanho razoavel sobre previdencia social. " * 4


def test_a_short_text_becomes_a_single_chunk():
    assert split_text("uma linha curta") == ["uma linha curta"]


def test_empty_and_blank_texts_produce_nothing():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_no_chunk_goes_over_the_limit():
    chunks = split_text(PARAGRAFO * 20)

    assert chunks
    assert all(len(chunk) <= CHUNK_CHARS for chunk in chunks)


def test_nothing_from_the_original_is_lost():
    # Cada caractere do original tem de aparecer em algum trecho. A sobreposicao
    # duplica parte, e isso e o desejado; perder e que nao pode.
    original = PARAGRAFO * 8
    chunks = split_text(original)

    emendado = "".join(chunks)
    for pedaco in original.split(". "):
        limpo = pedaco.strip()
        if limpo:
            assert limpo in emendado


def test_neighbours_overlap():
    # Texto com marcas unicas por posicao, para a sobreposicao ser verificavel: com
    # "abcabc..." qualquer pedaco casaria em qualquer lugar e o teste nao provaria
    # nada.
    texto = " ".join(f"marca{i:04d}" for i in range(400))
    chunks = split_text(texto, size=400, overlap=100)

    assert len(chunks) > 2
    for anterior, seguinte in pairwise(chunks):
        # O inicio do trecho seguinte tem de estar dentro do anterior -- e isso, e
        # so isso, que "sobreposicao" quer dizer.
        assert seguinte[:60] in anterior


def test_the_split_prefers_a_paragraph_boundary():
    primeiro = "a" * 300
    segundo = "b" * 300
    chunks = split_text(f"{primeiro}\n\n{segundo}", size=400, overlap=0)

    # O corte cai na quebra de paragrafo, entao o primeiro trecho e so os "a".
    assert chunks[0] == primeiro


def test_the_split_falls_back_to_a_sentence_boundary():
    texto = ("frase curta. " * 30).strip()
    chunks = split_text(texto, size=200, overlap=0)

    # Sem paragrafo disponivel, os cortes caem depois de um ponto.
    assert all(chunk.endswith(".") for chunk in chunks[:-1])


def test_accented_text_survives_intact():
    texto = "coração, atenção, pão, água, mãe, irmão, café, você, história. " * 40
    chunks = split_text(texto)

    assert "coração" in "".join(chunks)
    assert all("�" not in chunk for chunk in chunks)


def test_a_pathological_text_without_separators_still_terminates():
    # Nenhum espaco, nenhuma pontuacao: o corte cai no limite cru e o laco tem de
    # avancar de todo modo. Sem o `max(cut - overlap, 1)` isto travaria.
    chunks = split_text("x" * 5000, size=300, overlap=290)

    assert chunks
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_chunks_carry_the_source_name_and_their_position():
    source = Source(name="previdencia.txt", content=PARAGRAFO * 10)

    chunks = split_into_chunks(source)

    assert len(chunks) > 1
    assert all(chunk.source_name == "previdencia.txt" for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.total == len(chunks) for chunk in chunks)


def test_an_empty_source_yields_no_chunks():
    assert split_into_chunks(Source(name="vazio.txt", content="")) == []


def test_the_measured_defaults_stay_under_the_server_token_ceiling():
    # O servidor trunca em 511 tokens e medi ~3,9 chars/token em portugues, entao o
    # teto real fica perto de 2000 caracteres. Este teste existe para uma mudanca
    # descuidada em CHUNK_CHARS nao passar em silencio.
    assert CHUNK_CHARS / 3.9 < 511
    assert 0 < CHUNK_OVERLAP < CHUNK_CHARS
