from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest
from notebookgpt_core.models import Message
from notebookgpt_inference.client import InferenceError, LocalInferenceClient
from notebookgpt_inference.settings import InferenceSettings

SETTINGS = InferenceSettings(base_url="http://serv04.local:8000/v1", api_key="k", model="llama-3.2-1b")


def chunk(content: str | None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class FakeCompletions:
    """Um dublê de `AsyncOpenAI.chat.completions` que devolve, ou levanta."""

    def __init__(self, chunks=(), error: Exception | None = None):
        self._chunks = chunks
        self._error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error

        async def stream():
            for item in self._chunks:
                yield item

        return stream()


def fake_client(**kwargs):
    completions = FakeCompletions(**kwargs)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LocalInferenceClient(SETTINGS, client=client), completions


async def test_deltas_are_concatenated_in_order():
    client, _ = fake_client(chunks=[chunk("Olá"), chunk(", "), chunk("mundo")])

    assert await client.reply([Message(role="user", content="oi")]) == "Olá, mundo"


async def test_empty_and_missing_deltas_are_skipped():
    client, _ = fake_client(chunks=[chunk("a"), chunk(None), chunk(""), SimpleNamespace(choices=[]), chunk("b")])

    assert await client.reply([Message(role="user", content="oi")]) == "ab"


async def test_the_request_carries_the_model_streaming_and_the_messages():
    client, completions = fake_client(chunks=[chunk("x")])

    await client.reply([Message(role="system", content="s"), Message(role="user", content="u")])

    call = completions.calls[0]
    assert call["model"] == "llama-3.2-1b"
    assert call["stream"] is True
    assert call["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            openai.APIConnectionError(request=httpx.Request("POST", "http://serv04.local:8000/v1")),
            "Nao foi possivel falar com serv04.local:8000",
        ),
        (
            openai.AuthenticationError(
                "401",
                response=httpx.Response(401, request=httpx.Request("POST", "http://serv04.local:8000/v1")),
                body=None,
            ),
            "recusou a chave",
        ),
    ],
)
async def test_sdk_failures_become_an_inference_error_in_portuguese(error, expected):
    client, _ = fake_client(error=error)

    with pytest.raises(InferenceError, match=expected):
        await client.reply([Message(role="user", content="oi")])
