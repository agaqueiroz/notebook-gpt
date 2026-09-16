"""O banco vetorial embutido: uma tabela `vec0` do sqlite-vec sobre o sqlite3.

Por que nao precisa de metrica de cosseno nem de NumPy: o servidor devolve os
vetores ja normalizados (norma medida = 1.0), e para vetores unitarios a distancia
L2 padrao do vec0 ordena identicamente ao cosseno, porque L2^2 = 2 - 2*cos. O KNN
inteiro acontece em SQL.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import sqlite_vec
from notebookgpt_core import Chunk

EMBEDDING_DIMS = 384
"""As dimensoes do multilingual-e5-small, o modelo padrao.

Nao e uma trava do banco: a tabela nasce com o tamanho do primeiro vetor gravado.
O servidor ja trocou o modelo por baixo do mesmo nome (pedindo o e5, veio o
qwen3-embedding com 1024), e um indice de tamanho fixo derrubava o aplicativo.
"""

IN_MEMORY = ":memory:"


class VectorIndexError(Exception):
    """Falha ao abrir ou usar o banco vetorial."""


class DimensionMismatchError(VectorIndexError):
    """Um vetor com tamanho diferente do que o indice guarda.

    Quase sempre quer dizer que o modelo de embedding mudou: os vetores antigos e
    os novos nao estao no mesmo espaco, e o indice precisa ser refeito.
    """

    def __init__(self, expected: int, received: int):
        super().__init__(
            f"O indice guarda vetores de {expected} dimensoes e o servidor devolveu {received}. "
            "O modelo de embedding mudou; o indice precisa ser refeito."
        )
        self.expected = expected
        self.received = received


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
            self._db.execute("CREATE TABLE IF NOT EXISTS sources(name TEXT PRIMARY KEY, sha256 TEXT NOT NULL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            # Um `index.db` de antes do `meta` ja tem a tabela de trechos: o tamanho
            # sai da propria declaracao dela. O modelo fica desconhecido, o que faz
            # o Indexer refazer tudo uma vez -- nao da para saber quem gerou aqueles
            # vetores.
            if self._meta("dimensions") is None:
                declared = self._declared_dimensions()
                if declared is not None:
                    self._set_meta("dimensions", str(declared))

    def _create_chunks_table(self, dimensions: int) -> None:
        # `+content` e coluna auxiliar: o texto do trecho fica guardado sem ser
        # indexado, entao a busca ja devolve o conteudo sem uma segunda consulta.
        self._db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
                embedding float[{dimensions}],
                source_name text,
                ordinal integer,
                total integer,
                +content text
            )
            """
        )
        self._set_meta("dimensions", str(dimensions))

    def _declared_dimensions(self) -> int | None:
        row = self._db.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks'").fetchone()
        match = re.search(r"float\[(\d+)\]", row[0]) if row else None
        return int(match.group(1)) if match else None

    def _has_chunks_table(self) -> bool:
        return self._db.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunks'").fetchone() is not None

    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _check_dimensions(self, received: int) -> None:
        expected = self.dimensions
        if expected is not None and expected != received:
            raise DimensionMismatchError(expected, received)

    # -- leitura ---------------------------------------------------------------

    @property
    def dimensions(self) -> int | None:
        """Tamanho dos vetores guardados, ou None enquanto nada foi gravado."""
        value = self._meta("dimensions")
        return int(value) if value is not None else None

    @property
    def embedding_model(self) -> str | None:
        """O modelo que gerou os vetores guardados, ou None se desconhecido."""
        return self._meta("embedding_model")

    def indexed(self) -> dict[str, str]:
        """Nome da fonte -> sha256 do conteudo indexado."""
        return dict(self._db.execute("SELECT name, sha256 FROM sources").fetchall())

    def count(self) -> int:
        if not self._has_chunks_table():
            return 0
        return self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        """Os `k` trechos mais proximos, do mais proximo ao mais distante."""
        if k <= 0 or not self._has_chunks_table():
            return []
        vector = list(vector)
        self._check_dimensions(len(vector))
        try:
            rows = self._db.execute(
                "SELECT source_name, ordinal, total, content, distance FROM chunks WHERE embedding MATCH ? AND k = ?",
                (sqlite_vec.serialize_float32(vector), k),
            ).fetchall()
        except sqlite3.Error as error:
            raise VectorIndexError(f"Falha na busca do banco vetorial: {error}") from error
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
        """Troca todos os trechos de uma fonte pelos novos, numa transacao.

        Um vetor de tamanho diferente do indice levanta `DimensionMismatchError`
        antes de tocar o banco -- a fonte antiga continua la, intacta.
        """
        pairs = list(zip(chunks, vectors, strict=True))
        sizes = {len(vector) for _chunk, vector in pairs}
        if len(sizes) > 1:
            raise VectorIndexError(f"Vetores de tamanhos diferentes na mesma fonte: {sorted(sizes)}.")
        size = next(iter(sizes), None)
        if size is not None:
            self._check_dimensions(size)

        try:
            with self._db:
                if size is not None and not self._has_chunks_table():
                    self._create_chunks_table(size)
                if self._has_chunks_table():
                    self._db.execute("DELETE FROM chunks WHERE source_name = ?", (name,))
                # Fonte vazia antes de qualquer vetor: nao ha tabela ainda, e o
                # sqlite recusa preparar o INSERT mesmo sem linhas.
                if pairs:
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
                            for chunk, vector in pairs
                        ],
                    )
                self._db.execute(
                    "INSERT INTO sources(name, sha256) VALUES (?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET sha256 = excluded.sha256",
                    (name, sha256),
                )
        except sqlite3.Error as error:
            raise VectorIndexError(f"Falha ao gravar '{name}' no banco vetorial: {error}") from error

    def reset(self, embedding_model: str) -> None:
        """Esvazia o indice e o prepara para vetores de `embedding_model`.

        A tabela de trechos e apagada, nao so esvaziada: o tamanho dos vetores faz
        parte da declaracao dela, e o proximo modelo pode ter outro.
        """
        try:
            with self._db:
                self._db.execute("DROP TABLE IF EXISTS chunks")
                self._db.execute("DELETE FROM sources")
                self._db.execute("DELETE FROM meta WHERE key = 'dimensions'")
                self._set_meta("embedding_model", embedding_model)
        except sqlite3.Error as error:
            raise VectorIndexError(f"Falha ao refazer o banco vetorial: {error}") from error

    def drop_source(self, name: str) -> None:
        try:
            with self._db:
                if self._has_chunks_table():
                    self._db.execute("DELETE FROM chunks WHERE source_name = ?", (name,))
                self._db.execute("DELETE FROM sources WHERE name = ?", (name,))
        except sqlite3.Error as error:
            raise VectorIndexError(f"Falha ao remover '{name}' do banco vetorial: {error}") from error

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> VectorIndex:
        return self

    def __exit__(self, *_) -> None:
        self.close()
