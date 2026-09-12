import json

from notebookgpt_core.models import Message, NotebookData, Source
from notebookgpt_core.storage import load_notebook, notebook_from_text, notebook_to_text, save_notebook


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
