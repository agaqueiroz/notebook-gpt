"""As camadas do monorepo, verificadas no codigo-fonte.

`ui -> inference -> core`: o dominio nao conhece rede nem interface, e o cliente de
inferencia nao conhece a interface. Sem isso o PoC volta a ser um pacote achatado.
"""

import re
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[1] / "packages"

FORBIDDEN = {
    "notebookgpt-core": ("flet", "openai", "notebookgpt_inference", "notebookgpt_ui"),
    "notebookgpt-inference": ("flet", "notebookgpt_ui"),
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
