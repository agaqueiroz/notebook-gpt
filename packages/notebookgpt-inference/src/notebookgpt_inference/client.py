"""Camada anticorrupcao entre o NotebookGPT e o servidor local de inferencia.

O servidor implementa a API da OpenAI, entao o SDK oficial serve como cliente --
e e o que prova a compatibilidade. Nada acima desta camada importa `openai`:
daqui saem apenas trechos de texto e `InferenceError`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import cast

import openai
from notebookgpt_core.models import Message
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from notebookgpt_inference.settings import InferenceSettings

MAX_EMBEDDING_INPUTS = 64
"""Limite do servidor, medido: 65 entradas devolvem 400 `too_many_inputs`."""


class InferenceError(Exception):
    """Falha ao conversar com o servidor, com texto pronto para a interface."""


BUSY_CODE = "server_busy"
BUSY_MESSAGE = (
    "O servidor esta ocupado com outra inferencia. Ele atende um pedido por vez; "
    "espere a resposta atual terminar e tente de novo."
)


def _is_busy(error: openai.APIStatusError) -> bool:
    """O 503 de fila do servidor, que nao e uma falha real.

    O serv04 atende uma inferencia por vez e devolve 503 `server_busy` depois de
    LIS_BUSY_TIMEOUT. Dizer "respondeu com erro 503" manda o usuario procurar
    defeito onde ha so concorrencia.
    """
    if error.status_code != 503:
        return False
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("error")
    code = detail.get("code") if isinstance(detail, dict) else None
    return code == BUSY_CODE or BUSY_CODE in str(error)


def _wire_format(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    """So `role` e `content` vao para a API.

    `Message` tambem carrega os trechos que embasaram a resposta, para a interface
    mostrar a procedencia -- mas isso e assunto do caderno, nao do protocolo. Um
    `asdict` ingenuo mandaria `passages` junto e o servidor receberia um campo que
    nao existe na API da OpenAI.
    """
    wire = [{"role": message.role, "content": message.content} for message in messages]
    return cast("list[ChatCompletionMessageParam]", wire)


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
        # O modelo que o servidor declarou na ultima resposta de embedding. Pode
        # nao ser o pedido: o serv04 responde com o que tiver carregado.
        self.served_embedding_model: str | None = None

    async def stream_reply(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Emite a resposta em pedacos, na ordem em que o servidor os manda."""
        try:
            stream = await self._client.chat.completions.create(
                model=self.settings.model,
                messages=_wire_format(messages),
                temperature=self.settings.temperature,
                stream=True,
            )
            # Fechar explicitamente: sem isso a conexao SSE fica pendurada e o
            # loop de eventos reclama ao coletar o gerador ("generator didn't
            # stop after athrow()"). Nem todo dublê de teste tem `close`.
            closer = getattr(stream, "close", None)
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is not None and delta.content:
                        yield delta.content
            finally:
                if closer is not None:
                    await closer()
        except openai.AuthenticationError as error:
            raise InferenceError(
                f"{self.settings.host} recusou a chave. Confira NOTEBOOKGPT_API_KEY no seu .env."
            ) from error
        except openai.APIConnectionError as error:
            raise InferenceError(
                f"Nao foi possivel falar com {self.settings.host}. Verifique se o servidor esta no ar."
            ) from error
        except openai.APIStatusError as error:
            if _is_busy(error):
                raise InferenceError(BUSY_MESSAGE) from error
            raise InferenceError(f"{self.settings.host} respondeu com erro {error.status_code}.") from error
        except openai.OpenAIError as error:
            raise InferenceError(f"Falha ao conversar com {self.settings.host}: {error}") from error

    async def reply(self, messages: Sequence[Message]) -> str:
        """A resposta inteira, para quem nao precisa de streaming."""
        return "".join([chunk async for chunk in self.stream_reply(messages)])

    async def embed(self, texts: Sequence[str], *, prefix: str = "") -> list[list[float]]:
        """Os vetores de `texts`, na mesma ordem, em lotes que o servidor aceita.

        Os vetores chegam normalizados (norma medida = 1.0), entao quem consome
        pode tratar cosseno como produto escalar. `prefix` e o marcador do e5
        (`passage: ` ao indexar, `query: ` ao perguntar).
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_EMBEDDING_INPUTS):
            batch = [f"{prefix}{text}" for text in texts[start : start + MAX_EMBEDDING_INPUTS]]
            try:
                answer = await self._client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch,
                )
            except openai.AuthenticationError as error:
                raise InferenceError(
                    f"{self.settings.host} recusou a chave. Confira NOTEBOOKGPT_API_KEY no seu .env."
                ) from error
            except openai.APIConnectionError as error:
                raise InferenceError(
                    f"Nao foi possivel falar com {self.settings.host}. Verifique se o servidor esta no ar."
                ) from error
            except openai.APIStatusError as error:
                if _is_busy(error):
                    raise InferenceError(BUSY_MESSAGE) from error
                raise InferenceError(
                    f"Falha ao gerar embeddings em {self.settings.host}: erro {error.status_code}."
                ) from error
            except openai.OpenAIError as error:
                raise InferenceError(f"Falha ao gerar embeddings em {self.settings.host}: {error}") from error

            # `index` e o que garante a ordem: a API nao promete devolver os itens
            # na ordem de entrada, e um desalinhamento aqui associaria o vetor de um
            # trecho ao texto de outro -- um bug silencioso e dificil de ver.
            vectors.extend(item.embedding for item in sorted(answer.data, key=lambda item: item.index))
            self.served_embedding_model = getattr(answer, "model", None) or self.served_embedding_model

        return vectors

    async def aclose(self) -> None:
        """Fecha o pool de conexoes.

        Sem isso, o pool sobrevive ao fim do loop de eventos e o Python reclama ao
        coletar os geradores da httpx. Na interface o cliente vive o tempo do
        aplicativo; em script e teste, use o `async with`.
        """
        closer = getattr(self._client, "close", None)
        if closer is not None:
            await closer()

    async def __aenter__(self) -> LocalInferenceClient:
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()
