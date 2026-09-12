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


def test_reading_a_source_never_touches_the_original_file(tmp_path):
    """A garantia central do NotebookGPT: fonte de contexto e somente leitura.

    Verdadeiro por construcao hoje -- o seletor entrega bytes e o modulo so
    decodifica -- mas nada fixava isso. Aqui fica fixado.
    """
    path = tmp_path / "fonte.txt"
    path.write_text("conteudo original", encoding="utf-8")
    antes = (path.read_bytes(), path.stat().st_mtime_ns)

    read_txt_source(path)

    assert (path.read_bytes(), path.stat().st_mtime_ns) == antes


def test_a_source_in_a_read_only_folder_can_still_be_read(tmp_path):
    # O cenario do OneDrive: a pasta e somente leitura, e isso nao pode impedir
    # de usar o arquivo como contexto.
    folder = tmp_path / "protegida"
    folder.mkdir()
    path = folder / "fonte.txt"
    path.write_text("posso ser lido", encoding="utf-8")
    folder.chmod(0o500)
    try:
        assert read_txt_source(path).content == "posso ser lido"
    finally:
        folder.chmod(0o700)
