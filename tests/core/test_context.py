from notebookgpt_core.context import (
    HISTORY_TURNS,
    NO_SOURCES_NOTE,
    PER_SOURCE_CHARS,
    SYSTEM_PROMPT,
    TOTAL_SOURCE_CHARS,
    TRUNCATION_MARK,
    build_messages,
    format_sources,
)
from notebookgpt_core.models import Message, NotebookData, Source


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
