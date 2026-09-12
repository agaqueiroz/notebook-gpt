"""O RAG contra o serv04 de verdade — a comparação que justifica o trabalho.

uv run --env-file .env pytest -m network
"""

from collections.abc import AsyncIterator

import pytest
from notebookgpt_core import NotebookData, Source, build_messages, fits_whole
from notebookgpt_index import Indexer, VectorIndex
from notebookgpt_index.indexer import content_digest
from notebookgpt_inference import InferenceSettings, LocalInferenceClient

pytestmark = pytest.mark.network

BENEFICIOS = [
    ("aposentadoria por idade", "65 anos para homens e 62 anos para mulheres"),
    ("aposentadoria por tempo de contribuicao", "35 anos de contribuicao para homens"),
    ("auxilio por incapacidade temporaria", "carencia de 12 contribuicoes mensais"),
    ("salario-maternidade", "120 dias de duracao"),
    ("pensao por morte", "cota familiar de 50% mais 10% por dependente"),
    ("auxilio-acidente", "50% do salario de beneficio"),
    ("aposentadoria especial", "25 anos de exposicao a agentes nocivos"),
    ("beneficio de prestacao continuada", "renda per capita de um quarto do salario minimo"),
    ("auxilio-reclusao", "segurado de baixa renda recolhido a prisao"),
    ("abono anual", "pagamento junto ao beneficio de dezembro"),
    ("aposentadoria por incapacidade permanente", "pericia medica que ateste incapacidade total"),
    ("peculio", "devolucao de contribuicoes em hipoteses restritas"),
]

# Enchimento proprio por artigo, para cada um virar o seu trecho. Todos tem a mesma
# forma de proposito: distratores proximos e o que obriga o ranking a funcionar.
ENCHIMENTO = (
    "O requerimento e apresentado pelos canais remotos, com documentacao digitalizada, e a analise "
    "observa a ordem cronologica de entrada. A decisao e comunicada ao interessado, que pode "
    "apresentar recurso no prazo regulamentar as instancias revisoras. A concessao retroage a data "
    "de entrada do requerimento quando preenchidos os requisitos na origem. "
) * 2

REGULAMENTO = Source(
    name="regulamento.txt",
    content="\n\n".join(
        f"Artigo {i + 1}. O beneficio de {nome} exige {regra}. {ENCHIMENTO}"
        for i, (nome, regra) in enumerate(BENEFICIOS)
    ),
)

PERGUNTAS = [
    ("Qual a idade minima para aposentadoria por idade?", 1),
    ("Quantos anos de contribuicao para aposentadoria por tempo de contribuicao?", 2),
    ("Qual a carencia do auxilio por incapacidade temporaria?", 3),
    ("Quantos dias dura o salario-maternidade?", 4),
    ("Como se calcula a cota da pensao por morte?", 5),
    ("Qual o percentual do auxilio-acidente?", 6),
    ("Quantos anos de exposicao a agentes nocivos na aposentadoria especial?", 7),
    ("Qual a renda per capita exigida no beneficio de prestacao continuada?", 8),
    ("Quem tem direito ao auxilio-reclusao?", 9),
    ("Quando o abono anual e pago?", 10),
    ("O que a aposentadoria por incapacidade permanente exige?", 11),
    ("Em que hipoteses cabe peculio?", 12),
]

# Medi 12/12 em recall@1 neste corpus. O piso da folga para variacao do servidor e
# ainda derruba uma regressao de verdade: chunk mal dimensionado, prefixo dobrado ou
# a busca devolvendo qualquer coisa levam a taxa bem abaixo disso.
RECALL_MINIMO = 9


@pytest.fixture
def settings() -> InferenceSettings:
    loaded = InferenceSettings.from_env()
    if not loaded.api_key:
        pytest.skip("NOTEBOOKGPT_API_KEY não está no ambiente (veja a tabela de variáveis no README)")
    return loaded


@pytest.fixture
async def indexer(settings: InferenceSettings) -> AsyncIterator[Indexer]:
    # VectorIndex e sincrono (sqlite3); o cliente e assincrono.
    with VectorIndex(None) as store:
        async with LocalInferenceClient(settings) as client:
            pronto = Indexer(store, client)
            await pronto.sync([REGULAMENTO])
            yield pronto


def test_the_corpus_is_big_enough_to_need_retrieval():
    # Se o regulamento coubesse inteiro, o teste abaixo nao estaria medindo RAG.
    assert not fits_whole([REGULAMENTO])


async def test_every_article_becomes_its_own_chunk(indexer: Indexer):
    assert indexer.store.count() == len(BENEFICIOS)
    assert indexer.store.indexed() == {"regulamento.txt": content_digest(REGULAMENTO.content)}


async def test_retrieval_finds_the_article_that_answers_each_question(indexer: Indexer):
    acertos = []
    for pergunta, artigo in PERGUNTAS:
        marca = f"Artigo {artigo}."
        trechos = await indexer.retrieve(pergunta)
        posicoes = [i for i, trecho in enumerate(trechos) if marca in trecho.content]
        acertos.append((pergunta, marca, posicoes[0] if posicoes else None))

    no_topo = sum(posicao == 0 for _, _, posicao in acertos)
    relatorio = "\n".join(
        f"  [{'ok' if posicao == 0 else '--'}] {marca:<12} posicao={posicao} | {pergunta}"
        for pergunta, marca, posicao in acertos
    )
    assert no_topo >= RECALL_MINIMO, f"só {no_topo}/{len(PERGUNTAS)} no topo:\n{relatorio}"


async def test_a_buried_fact_reaches_the_model_only_through_retrieval(indexer: Indexer, settings: InferenceSettings):
    """A comparação que justifica o banco vetorial.

    O artigo 11 fica depois do corte de 6000 caracteres: pelo caminho antigo, de
    truncagem, ele simplesmente não chega ao modelo. Pela recuperação, chega.
    """
    pergunta = "O que a aposentadoria por incapacidade permanente exige?"
    notebook = NotebookData(sources=[REGULAMENTO])

    truncado = build_messages(notebook, pergunta)[-1].content
    assert "Artigo 11." not in truncado, "o corpus precisa enterrar o fato além da truncagem"

    trechos = await indexer.retrieve(pergunta)
    recuperado = build_messages(notebook, pergunta, trechos)[-1].content
    assert "Artigo 11." in recuperado
    assert "pericia medica" in recuperado

    # E o modelo, recebendo o trecho certo, responde com o fato.
    async with LocalInferenceClient(settings) as client:
        resposta = await client.reply(build_messages(notebook, pergunta, trechos))

    assert "pericia" in resposta.lower() or "perícia" in resposta.lower(), resposta


async def test_the_retrieved_context_stays_inside_the_budget(indexer: Indexer):
    trechos = await indexer.retrieve(PERGUNTAS[0][0])

    assert trechos
    assert sum(len(trecho.content) for trecho in trechos) <= 6000


async def test_reindexing_the_same_source_costs_nothing(indexer: Indexer):
    antes = indexer.store.indexed()

    assert await indexer.sync([REGULAMENTO]) == 0
    assert indexer.store.indexed() == antes
