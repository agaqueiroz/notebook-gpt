"""O banco vetorial embutido: uma tabela `vec0` do sqlite-vec sobre o sqlite3.

Por que nao precisa de metrica de cosseno nem de NumPy: o servidor devolve os
vetores ja normalizados (norma medida = 1.0), e para vetores unitarios a distancia
L2 padrao do vec0 ordena identicamente ao cosseno, porque L2^2 = 2 - 2*cos. O KNN
inteiro acontece em SQL.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import sqlite_vec
from notebookgpt_core import Chunk

EMBEDDING_DIMS = 384
"""Fixo: o multilingual-e5-small nao e Matryoshka e o servidor recusa `dimensions`."""

IN_MEMORY = ":memory:"


class VectorIndexError(Exception):
    """Falha ao abrir ou usar o banco vetorial."""


def _connect(path: Path | str) -> sqlite3.Connection:
    try:
        db = sqlite3.connect(path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    except (AttributeError, sqlite3.Error) as error:
        raise VectorIndexError(
            f"Nao foi possivel abrir o banco vetorial em {path}: {error}. "
            "A extensao sqlite-vec precisa de um Python com carregamento de extensoes habilitado."
        ) from error
    return db


class VectorIndex:
    """Os trechos das fontes e seus vetores, em um arquivo (ou em memoria).

    `VectorIndex(None)` abre em memoria -- e o caminho do modo web, onde nao existe
    diretorio para gravar; o indice e reconstruido a cada sessao.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = IN_MEMORY if path is None else path
        if self.path != IN_MEMORY:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = _connect(self.path)
        self._create_schema()

    def _create_schema(self) -> None:
        with self._db:
            # `+content` e coluna auxiliar: o texto do trecho fica guardado sem ser
            # indexado, entao a busca ja devolve o conteudo sem uma segunda consulta.
            self._db.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
                    embedding float[{EMBEDDING_DIMS}],
                    source_name text,
                    ordinal integer,
                    total integer,
                    +content text
                )
                """
            )
            self._db.execute("CREATE TABLE IF NOT EXISTS sources(name TEXT PRIMARY KEY, sha256 TEXT NOT NULL)")

    # -- leitura ---------------------------------------------------------------

    def indexed(self) -> dict[str, str]:
        """Nome da fonte -> sha256 do conteudo indexado."""
        return dict(self._db.execute("SELECT name, sha256 FROM sources").fetchall())

    def count(self) -> int:
        return self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        """Os `k` trechos mais proximos, do mais proximo ao mais distante."""
        if k <= 0:
            return []
        rows = self._db.execute(
            "SELECT source_name, ordinal, total, content, distance FROM chunks WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(list(vector)), k),
        ).fetchall()
        found = [
            (Chunk(source_name=name, ordinal=ordinal, total=total, content=content), distance)
            for name, ordinal, total, content, distance in rows
        ]
        # O vec0 ja devolve por distancia, mas a ordem e o contrato deste metodo:
        # quem chama preenche o orcamento na ordem do ranking.
        found.sort(key=lambda item: item[1])
        return found

    # -- escrita ---------------------------------------------------------------

    def replace_source(
        self,
        name: str,
        sha256: str,
        chunks: Iterable[Chunk],
        vectors: Iterable[Sequence[float]],
    ) -> None:
        """Troca todos os trechos de uma fonte pelos novos, numa transacao."""
        with self._db:
            self._db.execute("DELETE FROM chunks WHERE source_name = ?", (name,))
            self._db.executemany(
                "INSERT INTO chunks(embedding, source_name, ordinal, total, content) VALUES (?,?,?,?,?)",
                [
                    (
                        sqlite_vec.serialize_float32(list(vector)),
                        chunk.source_name,
                        chunk.ordinal,
                        chunk.total,
                        chunk.content,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            self._db.execute(
                "INSERT INTO sources(name, sha256) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET sha256 = excluded.sha256",
                (name, sha256),
            )

    def drop_source(self, name: str) -> None:
        with self._db:
            self._db.execute("DELETE FROM chunks WHERE source_name = ?", (name,))
            self._db.execute("DELETE FROM sources WHERE name = ?", (name,))

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> VectorIndex:
        return self

    def __exit__(self, *_) -> None:
        self.close()
