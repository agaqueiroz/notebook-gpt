"""As camadas do monorepo, verificadas no codigo-fonte.

`ui -> index -> inference -> core`: o dominio nao conhece rede, banco nem interface;
o cliente de inferencia nao conhece banco nem interface; o indice nao conhece a tela.
Sem isso o PoC volta a ser um pacote achatado.
"""

import re
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[1] / "packages"

FORBIDDEN = {
    "notebookgpt-core": (
        "flet",
        "openai",
        "sqlite_vec",
        "notebookgpt_inference",
        "notebookgpt_index",
        "notebookgpt_ui",
    ),
    "notebookgpt-inference": ("flet", "sqlite_vec", "notebookgpt_index", "notebookgpt_ui"),
    # O indice usa sqlite-vec e fala com o servidor via `inference`; nao conhece a tela.
    "notebookgpt-index": ("flet", "openai", "notebookgpt_ui"),
    "notebookgpt-ui": (),
}


def sources(package: str) -> list[Path]:
    found = sorted((PACKAGES / package / "src").rglob("*.py"))
    assert found, f"nenhum modulo encontrado em {package}"
    return found


def imported_root(line: str) -> str | None:
    """O pacote de topo importado por uma linha, ou None se nao for um import."""
    match = re.match(r"(?:from|import)\s+([A-Za-z_][\w.]*)", line.strip())
    return match.group(1).split(".", 1)[0] if match else None


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_packages_only_import_the_layers_below_them(package):
    banned = set(FORBIDDEN[package])
    for path in sources(package):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            root = imported_root(line)
            assert root not in banned, f"{path.name}:{number} importa {root}"
