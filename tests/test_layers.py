"""As camadas do monorepo, verificadas pelo import-linter.

`ui -> (index | inference) -> core`: o dominio nao conhece rede, banco nem interface;
o cliente de inferencia e o indice nao se conhecem nem conhecem a tela. Sem isso o
PoC volta a ser um pacote achatado.

Os contratos ficam em `[tool.importlinter]`, no pyproject da raiz. Rodar aqui faz
`uv run pytest` reprovar uma violacao mesmo sem CI.
"""

from pathlib import Path

from importlinter.cli import lint_imports

ROOT = Path(__file__).resolve().parents[1]


def test_import_contracts_are_kept(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert lint_imports() == 0
