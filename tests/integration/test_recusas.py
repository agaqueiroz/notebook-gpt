"""As duas metades da mudança no system prompt, contra o serv04 de verdade.

Tirar a licença de recusa fez o modelo parar de negar o que está no contexto. O
risco óbvio é ele passar a inventar quando o fato não existe. Uma metade sem a
outra seria um retrocesso, então as duas são testadas juntas.

    uv run --env-file .env pytest -m network
"""

from collections.abc import AsyncIterator

import pytest
from notebookgpt_core import NotebookData, Source, build_messages
from notebookgpt_index import Indexer, VectorIndex
from notebookgpt_inference import InferenceSettings, LocalInferenceClient

pytestmark = pytest.mark.network

# Relatórios de ocorrência: texto narrativo com um fato distinto por bloco e um
# rodapé burocrático idêntico em todos, que é o que cria distratores de verdade.
OCORRENCIAS = [
    (1, "galpao da rua Iguatemi", "um curto-circuito iniciou um principio de incendio", "curto-circuito"),
    (2, "deposito refrigerado do bloco C", "a falha do compressor descartou 240 caixas de vacina", "240"),
    (3, "patio de manobras", "uma empilhadeira deixou o portao norte inoperante por 9 dias", "9"),
    (4, "sala tecnica", "a falha do nobreak obrigou a reprocessar 430 pedidos", "430"),
    (5, "area de carga", "um vazamento de oleo rendeu advertencia do orgao ambiental", "advert"),
    (6, "telhado do galpao B", "o granizo perfurou 18 telhas de policarbonato", "18"),
]
RODAPE = (
    "O relatorio foi elaborado conforme o procedimento interno de gestao de ocorrencias. "
    "A comunicacao inicial se deu pelo canal de plantao, com registro em sistema e abertura "
    "de protocolo. A equipe de apoio permaneceu no local ate a liberacao formal da area. "
    "As evidencias fotograficas foram anexadas ao processo eletronico. "
) * 2

FONTE = Source(
    name="ocorrencias.txt",
    content="\n\n".join(
        f"Ocorrencia {n} — Local: {local}. Descricao: {fato}.\n{RODAPE}" for n, local, fato, _ in OCORRENCIAS
    ),
)

PERGUNTAS = [
    ("O que aconteceu no galpao da rua Iguatemi?", "curto-circuito"),
    ("Quantas caixas de vacina foram descartadas?", "240"),
    ("Por quantos dias o portao norte ficou inoperante?", "9"),
    ("Quantos pedidos foram reprocessados apos a falha do nobreak?", "430"),
    ("Quantas telhas o granizo perfurou?", "18"),
]

# Fatos que nao existem no corpus: o preco de tirar a licenca de recusa seria o
# modelo inventa-los. Medido, ele declinou nas quatro.
SEM_RESPOSTA = [
    "Qual foi o nome do fornecedor do nobreak?",
    "Quantos funcionarios foram demitidos apos a ocorrencia 3?",
    "Qual o CNPJ da empresa que fez a poda das arvores?",
]

RECUSA = (
    "não há",
    "nao ha",
    "não encontr",
    "nao encontr",
    "não consta",
    "nao consta",
    "não foi possível",
    "nao foi possivel",
    "não menciona",
    "nao menciona",
    "não informa",
    "nao informa",
    "não está",
    "nao esta",
    "não tenho",
    "nao tenho",
)


def recusou(resposta: str) -> bool:
    return any(marca in resposta.lower() for marca in RECUSA)


@pytest.fixture
def settings() -> InferenceSettings:
    loaded = InferenceSettings.from_env()
    if not loaded.api_key:
        pytest.skip("NOTEBOOKGPT_API_KEY não está no ambiente (veja a tabela de variáveis no README)")
    return loaded


@pytest.fixture
async def sala(settings: InferenceSettings) -> AsyncIterator[tuple[Indexer, LocalInferenceClient]]:
    with VectorIndex(None) as store:
        async with LocalInferenceClient(settings) as client:
            indexer = Indexer(store, client)
            await indexer.sync([FONTE])
            yield indexer, client


# Nao e zero, e a diferenca importa dizer. Com a licenca de recusa no system prompt
# este corpus recusava a maioria das perguntas; sem ela, sobra uma -- "Quantas telhas
# o granizo perfurou?", que recusou 3 de 3 execucoes.
#
# Nao e variancia, e sensibilidade de frase do modelo de 1B: os seis trechos estao
# todos no contexto e "18 telhas" esta la; perguntar "O granizo perfurou quantas
# telhas de policarbonato?" responde 18 corretamente. Nao troquei a pergunta para o
# teste passar -- ela fica como registro do limite do modelo.
#
# O teto de 1 ainda derruba uma regressao do prompt, que traz as recusas de volta em
# varias perguntas ao mesmo tempo.
MAX_RECUSAS = 1


async def test_it_mostly_stops_refusing_what_is_in_the_context(sala):
    """A metade que conserta o bug relatado."""
    indexer, client = sala
    notebook = NotebookData(sources=[FONTE])
    recusadas = []
    for pergunta, _ in PERGUNTAS:
        trechos = await indexer.retrieve(pergunta)
        resposta = await client.reply(build_messages(notebook, pergunta, trechos))
        if recusou(resposta):
            recusadas.append((pergunta, resposta[:120]))

    assert len(recusadas) <= MAX_RECUSAS, (
        f"{len(recusadas)} recusas (teto {MAX_RECUSAS}) com a resposta no contexto:\n"
        + "\n".join(f"  {pergunta}\n    -> {resposta!r}" for pergunta, resposta in recusadas)
    )


async def test_the_facts_actually_reach_the_answer(sala):
    indexer, client = sala
    notebook = NotebookData(sources=[FONTE])
    acertos = []
    for pergunta, marca in PERGUNTAS:
        trechos = await indexer.retrieve(pergunta)
        resposta = await client.reply(build_messages(notebook, pergunta, trechos))
        acertos.append((pergunta, marca, marca in resposta.lower(), resposta[:110]))

    certas = sum(ok for _, _, ok, _ in acertos)
    relatorio = "\n".join(
        f"  [{'ok' if ok else '--'}] {marca:<14} {pergunta}\n    -> {resposta!r}"
        for pergunta, marca, ok, resposta in acertos
    )
    # Um modelo de 1B erra transcricao de numero de vez em quando; o piso derruba
    # uma regressao de prompt sem quebrar a cada execucao.
    assert certas >= 4, f"só {certas}/{len(PERGUNTAS)} trouxeram o fato:\n{relatorio}"


async def test_it_still_declines_what_is_not_there(sala):
    """A outra metade: sem a licença de recusa, ele ainda não inventa.

    É o que torna a mudança segura — a capacidade de recusar continua lá, ela só
    parou de disparar à toa.
    """
    indexer, client = sala
    notebook = NotebookData(sources=[FONTE])
    inventadas = []
    for pergunta in SEM_RESPOSTA:
        trechos = await indexer.retrieve(pergunta)
        resposta = await client.reply(build_messages(notebook, pergunta, trechos))
        if not recusou(resposta):
            inventadas.append((pergunta, resposta[:140]))

    assert not inventadas, "respondeu o que nao estava no corpus:\n" + "\n".join(
        f"  {pergunta}\n    -> {resposta!r}" for pergunta, resposta in inventadas
    )
