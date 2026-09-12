from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest
from notebookgpt_core.models import Chunk, Message
from notebookgpt_inference.client import InferenceError, LocalInferenceClient
from notebookgpt_inference.settings import InferenceSettings

SETTINGS = InferenceSettings(base_url="http://serv04.local:8000/v1", api_key="k", model="llama-3.2-1b", temperature=0.0)


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


class FakeEmbeddings:
    """Dublê de `AsyncOpenAI.embeddings`, registrando cada lote pedido."""

    def __init__(self, error: Exception | None = None, shuffle: bool = False):
        self._error = error
        self._shuffle = shuffle
        self.batches: list[list[str]] = []

    async def create(self, **kwargs):
        batch = list(kwargs["input"])
        self.batches.append(batch)
        if self._error is not None:
            raise self._error
        # O primeiro componente e o indice, para os testes conferirem a ordem.
        data = [SimpleNamespace(index=i, embedding=[float(i)] + [0.0] * 383) for i in range(len(batch))]
        if self._shuffle:
            data = list(reversed(data))
        return SimpleNamespace(data=data)


def fake_client_with_embeddings(*, count: int = 0, shuffle: bool = False, error: Exception | None = None):
    embeddings = FakeEmbeddings(error=error, shuffle=shuffle)
    stub = SimpleNamespace(embeddings=embeddings)
    return LocalInferenceClient(SETTINGS, client=stub), embeddings


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
    assert call["temperature"] == 0.0
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


def test_the_api_key_never_shows_up_in_the_settings_repr():
    # Um traceback de pytest imprime as fixtures; a chave nao pode ir junto.
    settings = InferenceSettings(api_key="chave-secreta-de-verdade")

    assert "chave-secreta-de-verdade" not in repr(settings)
    assert "serv04.local" in repr(settings)


def test_numeric_settings_fall_back_when_the_environment_is_garbage():
    settings = InferenceSettings.from_env({"NOTEBOOKGPT_TIMEOUT": "nao e numero", "NOTEBOOKGPT_TEMPERATURE": ""})

    assert settings.timeout == 120.0
    assert settings.temperature == 0.0


async def test_embeddings_are_split_into_batches_the_server_accepts():
    # O servidor recusa mais de 64 entradas por requisicao (medido: 400
    # `too_many_inputs` em 65). A divisao mora aqui, e nao em quem chama.
    client, embeddings = fake_client_with_embeddings(count=150)

    vectors = await client.embed([f"trecho {i}" for i in range(150)])

    assert len(vectors) == 150
    assert [len(batch) for batch in embeddings.batches] == [64, 64, 22]


async def test_embeddings_come_back_in_the_order_they_went_in():
    # A API nao promete a ordem: um desalinhamento associaria o vetor de um trecho
    # ao texto de outro, e o erro seria invisivel.
    client, _ = fake_client_with_embeddings(count=3, shuffle=True)

    vectors = await client.embed(["a", "b", "c"])

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]


async def test_the_prefix_is_applied_to_every_input():
    client, embeddings = fake_client_with_embeddings(count=2)

    await client.embed(["um", "dois"], prefix="passage: ")

    assert embeddings.batches[0] == ["passage: um", "passage: dois"]


async def test_embedding_no_texts_asks_the_server_nothing():
    client, embeddings = fake_client_with_embeddings(count=0)

    assert await client.embed([]) == []
    assert embeddings.batches == []


async def test_an_embedding_failure_becomes_an_inference_error():
    client, _ = fake_client_with_embeddings(
        count=1,
        error=openai.APIConnectionError(request=httpx.Request("POST", "http://serv04.local:8000/v1")),
    )

    with pytest.raises(InferenceError, match=r"Nao foi possivel falar com serv04\.local:8000"):
        await client.embed(["um"])


async def test_the_passages_of_a_message_never_reach_the_api():
    # `Message` carrega os trechos que embasaram a resposta, para a interface
    # mostrar a procedencia. Mandar isso no payload seria um campo inexistente na
    # API da OpenAI.
    client, completions = fake_client(chunks=[chunk("x")])
    com_trechos = Message(
        role="user",
        content="u",
        passages=[Chunk(source_name="a.txt", ordinal=1, total=3, content="trecho")],
    )

    await client.reply([com_trechos])

    assert completions.calls[0]["messages"] == [{"role": "user", "content": "u"}]
