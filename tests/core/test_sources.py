import pytest
from notebookgpt_core.models import Source
from notebookgpt_core.sources import UnsupportedSourceError, is_supported, read_txt_source, source_from_bytes


def test_reads_a_txt_file_from_disk(tmp_path):
    path = tmp_path / "notas.txt"
    path.write_text("linha um\nlinha dois", encoding="utf-8")

    assert read_txt_source(path) == Source(name="notas.txt", content="linha um\nlinha dois")


@pytest.mark.parametrize("name", ["relatorio.pdf", "planilha.xlsx", "sem-extensao", "codigo.py"])
def test_rejects_everything_that_is_not_txt(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"qualquer coisa")

    assert not is_supported(name)
    with pytest.raises(UnsupportedSourceError):
        read_txt_source(path)
    with pytest.raises(UnsupportedSourceError):
        source_from_bytes(name, b"qualquer coisa")


def test_the_txt_check_ignores_letter_case(tmp_path):
    path = tmp_path / "GRITANDO.TXT"
    path.write_text("ok", encoding="utf-8")

    assert read_txt_source(path).content == "ok"


def test_undecodable_bytes_are_replaced_instead_of_failing():
    source = source_from_bytes("latin.txt", b"cora\xe7\xe3o")

    assert source.name == "latin.txt"
    assert source.content.startswith("cora")
