from notebookgpt_core.context import (
    HISTORY_TURNS,
    NO_SOURCES_NOTE,
    PER_SOURCE_CHARS,
    SYSTEM_PROMPT,
    TOTAL_SOURCE_CHARS,
    TRUNCATION_MARK,
    build_messages,
    fits_whole,
    format_passages,
    format_sources,
)
from notebookgpt_core.models import Chunk, Message, NotebookData, Source


def test_only_the_instructions_go_in_system():
    request = build_messages(NotebookData(), "qual a capital?")

    assert request[0] == Message(role="system", content=SYSTEM_PROMPT)
    assert [message.role for message in request].count("system") == 1


def test_the_sources_travel_with_the_prompt_in_the_last_user_turn():
    # Medido contra o llama-3.2-1b: fontes em `system` sao ignoradas pelo modelo.
    # Este teste e o que impede a regressao de voltar a coloca-las la.
    notebook = NotebookData(sources=[Source(name="receita.txt", content="bata as claras")])

    request = build_messages(notebook, "qual a capital?")

    assert request[-1].role == "user"
    assert "qual a capital?" in request[-1].content
    assert "receita.txt" in request[-1].content
    assert "bata as claras" in request[-1].content


def test_without_sources_the_model_is_told_so():
    request = build_messages(NotebookData(), "oi")

    assert NO_SOURCES_NOTE in request[-1].content


def test_source_names_and_content_reach_the_model():
    notebook = NotebookData(sources=[Source(name="receita.txt", content="bata as claras")])

    block = format_sources(notebook.sources)

    assert "receita.txt" in block
    assert "bata as claras" in block


# Um caractere que nao aparece em nenhum rotulo do bloco montado ("Fontes
# anexadas", ".txt", "[fonte truncada]"), para poder contar so o conteudo.
FILLER = "z"


def test_a_long_source_is_truncated_with_a_visible_mark():
    block = format_sources([Source(name="grande.txt", content=FILLER * (PER_SOURCE_CHARS * 2))])

    assert TRUNCATION_MARK in block
    assert block.count(FILLER) == PER_SOURCE_CHARS


def test_many_sources_stay_inside_the_total_budget():
    sources = [Source(name=f"f{i}.txt", content=FILLER * PER_SOURCE_CHARS) for i in range(10)]

    block = format_sources(sources)

    assert block.count(FILLER) <= TOTAL_SOURCE_CHARS
    # Toda fonte continua aparecendo, mesmo as que nao couberam: o usuario precisa
    # ver que elas foram cortadas.
    for source in sources:
        assert source.name in block


def test_only_the_most_recent_turns_are_replayed():
    history = [Message(role="user", content=str(i)) for i in range(HISTORY_TURNS * 3)]

    request = build_messages(NotebookData(messages=history), "agora")

    replayed = [message for message in request if message.role != "system"][:-1]
    assert len(replayed) == HISTORY_TURNS
    assert replayed[-1].content == history[-1].content
    # O historico entra cru: as fontes so acompanham o turno atual.
    assert all("Fontes anexadas" not in message.content for message in replayed)


def test_retrieved_passages_keep_the_measured_format_and_position():
    # A medicao que vale para as fontes inteiras vale para os trechos: texto
    # delimitado, no turno do usuario. O RAG nao e desculpa para mudar o formato.
    passages = [Chunk(source_name="a.txt", ordinal=2, total=7, content="bata as claras")]

    request = build_messages(NotebookData(), "como faco?", passages)

    assert [message.role for message in request].count("system") == 1
    assert request[-1].role == "user"
    assert '"""' in request[-1].content
    assert "##" not in request[-1].content
    assert "como faco?" in request[-1].content


def test_a_passage_says_which_file_and_which_piece_it_came_from():
    passages = [Chunk(source_name="previdencia.txt", ordinal=3, total=9, content="texto")]

    block = format_passages(passages)

    assert 'Arquivo "previdencia.txt" (trecho 3 de 9):' in block


def test_passages_replace_the_whole_sources_when_given():
    notebook = NotebookData(sources=[Source(name="grande.txt", content="conteudo inteiro da fonte")])
    passages = [Chunk(source_name="grande.txt", ordinal=1, total=4, content="so este trecho")]

    request = build_messages(notebook, "pergunta", passages)

    assert "so este trecho" in request[-1].content
    assert "conteudo inteiro da fonte" not in request[-1].content


def test_an_empty_retrieval_tells_the_model_there_is_nothing():
    request = build_messages(NotebookData(), "pergunta", [])

    assert NO_SOURCES_NOTE in request[-1].content


def test_fits_whole_decides_by_the_same_budget_the_context_uses():
    assert fits_whole([])
    assert fits_whole([Source(name="a.txt", content="x" * TOTAL_SOURCE_CHARS)])
    assert not fits_whole([Source(name="a.txt", content="x" * (TOTAL_SOURCE_CHARS + 1))])
    # A soma e o que conta, nao o tamanho de cada uma.
    metade = "x" * (TOTAL_SOURCE_CHARS // 2)
    assert fits_whole([Source(name="a.txt", content=metade), Source(name="b.txt", content=metade)])
    assert not fits_whole([Source(name=f"{i}.txt", content=metade) for i in range(3)])
