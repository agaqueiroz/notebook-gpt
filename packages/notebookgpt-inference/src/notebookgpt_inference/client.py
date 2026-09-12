"""Camada anticorrupcao entre o NotebookGPT e o servidor local de inferencia.

O servidor implementa a API da OpenAI, entao o SDK oficial serve como cliente --
e e o que prova a compatibilidade. Nada acima desta camada importa `openai`:
daqui saem apenas trechos de texto e `InferenceError`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict

import openai
from notebookgpt_core.models import Message
from openai import AsyncOpenAI

from notebookgpt_inference.settings import InferenceSettings


class InferenceError(Exception):
    """Falha ao conversar com o servidor, com texto pronto para a interface."""


def _build_client(settings: InferenceSettings) -> AsyncOpenAI:
    # O SDK recusa uma chave vazia antes de qualquer requisicao; o placeholder deixa
    # o servidor responder 401 e a mensagem de erro vir dele, nao do cliente.
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key or "sem-chave",
        timeout=settings.timeout,
        max_retries=1,
    )


class LocalInferenceClient:
    """Conversa com o servidor local; `client` e injetavel para os testes."""

    def __init__(self, settings: InferenceSettings | None = None, client: AsyncOpenAI | None = None):
        self.settings = settings or InferenceSettings.from_env()
        self._client = client or _build_client(self.settings)

    async def stream_reply(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Emite a resposta em pedacos, na ordem em que o servidor os manda."""
        try:
            stream = await self._client.chat.completions.create(
                model=self.settings.model,
                messages=[asdict(message) for message in messages],
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is not None and delta.content:
                    yield delta.content
        except openai.AuthenticationError as error:
            raise InferenceError(
                f"{self.settings.host} recusou a chave. Confira NOTEBOOKGPT_API_KEY no seu .env."
            ) from error
        except openai.APIConnectionError as error:
            raise InferenceError(
                f"Nao foi possivel falar com {self.settings.host}. Verifique se o servidor esta no ar."
            ) from error
        except openai.APIStatusError as error:
            raise InferenceError(f"{self.settings.host} respondeu com erro {error.status_code}.") from error
        except openai.OpenAIError as error:
            raise InferenceError(f"Falha ao conversar com {self.settings.host}: {error}") from error

    async def reply(self, messages: Sequence[Message]) -> str:
        """A resposta inteira, para quem nao precisa de streaming."""
        return "".join([chunk async for chunk in self.stream_reply(messages)])
