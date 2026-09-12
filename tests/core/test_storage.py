import json

import pytest
from notebookgpt_core.models import Chunk, Message, NotebookData, Source
from notebookgpt_core.storage import (
    StorageError,
    load_notebook,
    notebook_from_text,
    notebook_to_text,
    save_notebook,
)


def test_round_trip_preserves_sources_messages_and_notes():
    notebook = NotebookData(
        sources=[Source(name="a.txt", content="conteudo de a"), Source(name="b.txt", content="")],
        messages=[Message(role="user", content="oi"), Message(role="assistant", content="ola")],
        notes="minhas anotacoes",
    )

    restored = notebook_from_text(notebook_to_text(notebook))

    assert restored == notebook


def test_legacy_sources_as_plain_names_still_load():
    legacy = json.dumps({"sources": ["antigo.txt"], "messages": [], "notes": ""})

    restored = notebook_from_text(legacy)

    assert restored.sources == [Source(name="antigo.txt", content="")]


def test_garbage_and_empty_text_fall_back_to_an_empty_notebook():
    assert notebook_from_text(None) == NotebookData()
    assert notebook_from_text("") == NotebookData()
    assert notebook_from_text("nao e json") == NotebookData()
    assert notebook_from_text("[1, 2, 3]") == NotebookData()


def test_save_creates_missing_directories_and_load_reads_it_back(tmp_path):
    path = tmp_path / "sub" / "notebookgpt.txt"
    notebook = NotebookData(sources=[Source(name="a.txt", content="x")], notes="n")

    save_notebook(path, notebook)

    assert load_notebook(path) == notebook


def test_loading_a_missing_file_yields_an_empty_notebook(tmp_path):
    assert load_notebook(tmp_path / "nao-existe.txt") == NotebookData()


def test_a_blocked_write_becomes_a_storage_error_with_the_path(tmp_path):
    # Um arquivo no lugar da pasta: o mkdir falha com OSError, como a Protecao
    # contra Ransomware faz na pratica -- sem precisar mexer no Windows.
    blocker = tmp_path / "nao-e-pasta"
    blocker.write_text("sou um arquivo", encoding="utf-8")
    path = blocker / "notebookgpt.txt"

    with pytest.raises(StorageError) as caught:
        save_notebook(path, NotebookData(notes="n"))

    mensagem = str(caught.value)
    assert str(path) in mensagem
    assert "Ransomware" in mensagem


def test_the_storage_error_keeps_the_original_cause(tmp_path):
    blocker = tmp_path / "nao-e-pasta"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(StorageError) as caught:
        save_notebook(blocker / "notebookgpt.txt", NotebookData())

    assert isinstance(caught.value.__cause__, OSError)


def test_the_passages_behind_an_answer_survive_a_round_trip():
    notebook = NotebookData(
        messages=[
            Message(role="user", content="qual a regra?"),
            Message(
                role="assistant",
                content="A regra e X.",
                passages=[
                    Chunk(source_name="regulamento.txt", ordinal=3, total=9, content="texto do trecho"),
                    Chunk(source_name="anexo.txt", ordinal=1, total=2, content="outro trecho"),
                ],
            ),
        ]
    )

    restored = notebook_from_text(notebook_to_text(notebook))

    assert restored == notebook


def test_messages_written_before_the_vector_index_still_load():
    legacy = json.dumps({"sources": [], "messages": [{"role": "user", "content": "oi"}], "notes": ""})

    restored = notebook_from_text(legacy)

    assert restored.messages == [Message(role="user", content="oi", passages=[])]


def test_a_message_missing_its_role_does_not_crash_the_load():
    # Caderno corrompido abre vazio em vez de derrubar o aplicativo.
    quebrado = json.dumps({"messages": [{"content": "sem papel"}]})

    assert notebook_from_text(quebrado) == NotebookData()


def test_a_passage_with_a_bad_ordinal_does_not_crash_the_load():
    quebrado = json.dumps(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "x",
                    "passages": [{"source_name": "a.txt", "ordinal": "nao e numero", "total": 2, "content": "t"}],
                }
            ]
        }
    )

    assert notebook_from_text(quebrado) == NotebookData()
