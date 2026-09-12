"""Configuracao do servidor local de inferencia, lida do ambiente."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://serv04.local:8000/v1"
DEFAULT_MODEL = "llama-3.2-1b"
DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class InferenceSettings:
    """Onde falar, com que chave e com qual modelo.

    A chave nunca tem valor padrao no codigo: ela vem de NOTEBOOKGPT_API_KEY
    (um `.env` local, fora do controle de versao) ou o pedido falha.
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> InferenceSettings:
        env = os.environ if env is None else env
        try:
            timeout = float(env.get("NOTEBOOKGPT_TIMEOUT", DEFAULT_TIMEOUT))
        except ValueError:
            timeout = DEFAULT_TIMEOUT
        return cls(
            base_url=env.get("NOTEBOOKGPT_BASE_URL") or DEFAULT_BASE_URL,
            api_key=env.get("NOTEBOOKGPT_API_KEY", ""),
            model=env.get("NOTEBOOKGPT_MODEL") or DEFAULT_MODEL,
            timeout=timeout,
        )

    @property
    def host(self) -> str:
        """So para mensagens de erro: 'serv04.local:8000' em vez da URL inteira."""
        return self.base_url.split("//", 1)[-1].split("/", 1)[0]
