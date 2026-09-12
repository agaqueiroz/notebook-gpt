"""Conversa de verdade com o servidor local.

Marcado como `network` e fora da seleção padrão: precisa do serv04.local no ar e
de `NOTEBOOKGPT_API_KEY` no ambiente. É o teste que prova o PoC de ponta a ponta.

    uv run --env-file .env pytest -m network
"""

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest
from notebookgpt_core import NotebookData, Source, build_messages
from notebookgpt_inference import InferenceError, InferenceSettings, LocalInferenceClient

pytestmark = pytest.mark.network


@pytest.fixture
def settings() -> InferenceSettings:
    loaded = InferenceSettings.from_env()
    if not loaded.api_key:
        pytest.skip("NOTEBOOKGPT_API_KEY não está no ambiente (veja .env.example)")
    return loaded


@pytest.fixture
async def client(settings: InferenceSettings) -> AsyncIterator[LocalInferenceClient]:
    async with LocalInferenceClient(settings) as connected:
        yield connected


# Fatos que o modelo não pode saber de outro lugar: se aparecem na resposta,
# vieram da fonte anexada.
NOTEBOOK = NotebookData(
    sources=[
        Source(
            name="casa-da-ana.txt",
            content=(
                "A casa da Ana fica na rua das Acacias, numero 47.\n"
                "O telhado da casa da Ana e verde.\n"
                "O portao e de madeira e a caixa de correio e amarela."
            ),
        )
    ]
)

PERGUNTAS = [
    ("De que cor e o telhado?", "verde"),
    ("Qual o numero da casa?", "47"),
    ("Em que rua fica a casa?", "acacias"),
    ("De que material e o portao?", "madeira"),
    ("Qual a cor da caixa de correio?", "amarela"),
]

# O llama-3.2-1b não acerta as cinco. Com temperature=0 e o formato de fonte
# atual ele fica em 4/5; o piso de 3 dá folga para variação do servidor e ainda
# derruba uma regressão de verdade -- trocar o texto delimitado por cabeçalhos
# Markdown, por exemplo, leva a taxa para 1-2/5.
GROUNDING_MINIMO = 3


async def test_the_answers_come_from_the_attached_source(client: LocalInferenceClient):
    acertos = []
    for pergunta, esperado in PERGUNTAS:
        answer = await client.reply(build_messages(NOTEBOOK, pergunta))
        acertos.append((pergunta, esperado, esperado in answer.lower(), answer))

        assert answer.strip(), f"resposta vazia para {pergunta!r}"

    fundamentadas = sum(ok for _, _, ok, _ in acertos)
    relatorio = "\n".join(
        f"  [{'ok' if ok else '--'}] {pergunta} (esperava {esperado!r}) -> {answer[:90]!r}"
        for pergunta, esperado, ok, answer in acertos
    )
    assert fundamentadas >= GROUNDING_MINIMO, f"só {fundamentadas}/{len(PERGUNTAS)} fundamentadas:\n{relatorio}"


async def test_the_answer_really_arrives_in_pieces(client: LocalInferenceClient):
    request = build_messages(NOTEBOOK, "Descreva a casa da Ana.")

    chunks = [chunk async for chunk in client.stream_reply(request)]

    assert len(chunks) > 1, f"esperava streaming, veio tudo de uma vez: {chunks}"
    assert "".join(chunks).strip()


async def test_the_model_is_told_when_there_are_no_sources(client: LocalInferenceClient):
    answer = await client.reply(build_messages(NotebookData(), "Diga apenas: ok."))

    assert answer.strip()


async def test_a_wrong_key_is_refused_by_the_real_server(settings: InferenceSettings):
    async with LocalInferenceClient(replace(settings, api_key="chave-errada")) as client:
        with pytest.raises(InferenceError, match="recusou a chave"):
            await client.reply(build_messages(NotebookData(), "oi"))


async def test_an_unknown_model_alias_is_silently_ignored_by_the_server(settings: InferenceSettings):
    """Documenta um comportamento do servidor, não do cliente.

    O serv04 carrega um modelo só e não valida o alias pedido: um erro de
    digitação em `NOTEBOOKGPT_MODEL` responde normalmente, com o modelo
    carregado, em vez de falhar. Vale saber ao depurar "por que a resposta não
    mudou quando troquei de modelo". Se o servidor passar a validar, este teste
    falha e avisa.
    """
    async with LocalInferenceClient(replace(settings, model="modelo-que-nao-existe")) as client:
        answer = await client.reply(build_messages(NotebookData(), "Diga apenas: ok."))

    assert answer.strip()
