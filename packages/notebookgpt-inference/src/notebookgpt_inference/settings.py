"""Configuracao do servidor local de inferencia, lida do ambiente."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "http://serv04.local:8000/v1"
DEFAULT_MODEL = "llama-3.2-1b"
DEFAULT_TIMEOUT = 120.0
# Zero por padrao: a tarefa e responder a partir de uma fonte anexada, nao
# inventar. Tambem torna a resposta reproduzivel, o que os testes de rede usam.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_EMBEDDING_MODEL = "multilingual-e5-small"
# O e5 foi treinado com prefixos distintos para documento e pergunta, e sem eles
# a qualidade de recuperacao cai. O servidor aplica um prefixo unico a toda
# entrada (LIS_EMBEDDING_INPUT_PREFIX); com ele esvaziado la, estes valem aqui.
# Se o servidor ainda estiver prefixando, esvazie estes para nao dobrar.
DEFAULT_PASSAGE_PREFIX = "passage: "
DEFAULT_QUERY_PREFIX = "query: "


@dataclass(frozen=True)
class InferenceSettings:
    """Onde falar, com que chave e com qual modelo.

    A chave nunca tem valor padrao no codigo: ela vem de NOTEBOOKGPT_API_KEY
    (um `.env` local, fora do controle de versao) ou o pedido falha.
    """

    base_url: str = DEFAULT_BASE_URL
    # Fora do repr: sem isso a chave vaza no output do pytest, em traceback e
    # em qualquer log que imprima as configuracoes.
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    passage_prefix: str = DEFAULT_PASSAGE_PREFIX
    query_prefix: str = DEFAULT_QUERY_PREFIX

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> InferenceSettings:
        env = os.environ if env is None else env

        def number(name: str, default: float) -> float:
            try:
                return float(env.get(name, default))
            except ValueError:
                return default

        return cls(
            base_url=env.get("NOTEBOOKGPT_BASE_URL") or DEFAULT_BASE_URL,
            api_key=env.get("NOTEBOOKGPT_API_KEY", ""),
            model=env.get("NOTEBOOKGPT_MODEL") or DEFAULT_MODEL,
            timeout=number("NOTEBOOKGPT_TIMEOUT", DEFAULT_TIMEOUT),
            temperature=number("NOTEBOOKGPT_TEMPERATURE", DEFAULT_TEMPERATURE),
            embedding_model=env.get("NOTEBOOKGPT_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
            # `or` nao serve aqui: string vazia e uma escolha valida e precisa
            # sobreviver, e o jeito de desligar o prefixo do cliente.
            passage_prefix=env.get("NOTEBOOKGPT_PASSAGE_PREFIX", DEFAULT_PASSAGE_PREFIX),
            query_prefix=env.get("NOTEBOOKGPT_QUERY_PREFIX", DEFAULT_QUERY_PREFIX),
        )

    @property
    def host(self) -> str:
        """So para mensagens de erro: 'serv04.local:8000' em vez da URL inteira."""
        return self.base_url.split("//", 1)[-1].split("/", 1)[0]
