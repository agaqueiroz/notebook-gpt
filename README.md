# NotebookGPT

Um workspace escuro para conversas fundamentadas em arquivos de texto e anotações
pessoais. **É um PoC** (prova de conceito): existe para mostrar o caminho completo
— fonte TXT → contexto → modelo local → resposta em Markdown na tela — e não para
ser usado em produção.

As conversas acontecem contra um servidor de inferência **local**, que implementa a
API da OpenAI: [agaqueiroz/local-inference-server](https://github.com/agaqueiroz/local-inference-server).
Nenhum prompt sai da rede de casa.

## Como rodar

```bash
uv sync
cp .env.example .env     # e preencha NOTEBOOKGPT_API_KEY
uv run notebookgpt
```

A chave é a mesma `LIS_API_KEY` configurada no servidor. O `.env` não é versionado.

| Variável | Padrão | O que é |
| --- | --- | --- |
| `NOTEBOOKGPT_BASE_URL` | `http://serv04.local:8000/v1` | Endereço do servidor |
| `NOTEBOOKGPT_API_KEY` | *(vazio)* | Chave exigida pelo servidor |
| `NOTEBOOKGPT_MODEL` | `llama-3.2-1b` | Modelo de chat |
| `NOTEBOOKGPT_TEMPERATURE` | `0` | Zero para responder a partir da fonte, não inventar |
| `NOTEBOOKGPT_TIMEOUT` | `120` | Tempo limite, em segundos |

Para conferir se o servidor está no ar antes de abrir a janela:

```bash
curl -s http://serv04.local:8000/health
```

## O monorepo

Um workspace [uv](https://docs.astral.sh/uv/concepts/projects/workspaces/) com um
`.venv` e um `uv.lock` na raiz, e três membros em `packages/`:

| Pacote | Responsabilidade | Depende de |
| --- | --- | --- |
| `notebookgpt-core` | Caderno, fontes TXT, persistência e montagem do contexto | biblioteca padrão |
| `notebookgpt-inference` | Cliente do servidor local, streaming e tradução de erros | `core`, `openai` |
| `notebookgpt-ui` | Interface Flet (o pacote original deste repositório) | `core`, `inference`, `flet` |

A direção é sempre `ui → inference → core`. O domínio não conhece rede nem
interface, e o cliente de inferência não conhece a interface — é o que permite
testar a montagem do contexto e o streaming sem abrir uma janela.
[tests/test_layers.py](tests/test_layers.py) verifica isso no código-fonte.

## Fontes de contexto

Nesta etapa, **somente `.txt`**. O seletor de arquivos já filtra por extensão e
`notebookgpt_core.sources` recusa o resto com `UnsupportedSourceError`.

As fontes são lidas no momento em que você as adiciona e ficam gravadas junto do
caderno — é o que faz elas sobreviverem a um reinício também no modo web, onde o
Flet nunca devolve o caminho do arquivo. Como o servidor roda com
`LIS_MAX_MODEL_LENGTH=4096`, `notebookgpt_core.context` corta as fontes por
orçamento de caracteres e marca o corte com `… [fonte truncada]`. É a limitação
mais visível do PoC; [TODO.md](TODO.md) aponta o caminho do RAG.

### O formato do contexto não é cosmético

Duas decisões de `notebookgpt_core.context` foram **medidas** contra o
`llama-3.2-1b` do serv04, com `temperature=0`, em cinco perguntas cuja resposta
está na fonte anexada:

| Formato das fontes | Onde elas entram | Respostas fundamentadas |
| --- | --- | --- |
| Cabeçalhos Markdown (`## Fontes anexadas`) | `system` | 0/5 — "não tenho acesso a essa informação" |
| Cabeçalhos Markdown | turno do usuário | 2/5 — chega a dizer "não posso responder" |
| Texto delimitado (`Arquivo "x.txt": """…"""`) | turno do usuário | **4/5** |

Um modelo pequeno trata o cabeçalho Markdown como conteúdo a interpretar, não
como moldura, e ignora o que está em `system`. Por isso as fontes vão delimitadas,
dentro da mensagem do usuário, junto da pergunta.
[tests/integration/test_serv04.py](tests/integration/test_serv04.py) fixa um piso
de 3/5 para que uma regressão nesse formato apareça.

## Desenvolvimento

```bash
uv run ruff check
uv run ruff format
uv run pytest              # rápido, sem rede
uv run pytest -m network   # conversa de verdade com o serv04.local
```

A suíte padrão não toca a rede: o cliente aceita um `AsyncOpenAI` injetado
([tests/inference/test_client.py](tests/inference/test_client.py)).

Os testes marcados `network` ficam fora da seleção padrão e são o que prova o PoC
de ponta a ponta — resposta fundamentada na fonte anexada, streaming em pedaços e
chave errada recusada pelo servidor de verdade
([tests/integration/test_serv04.py](tests/integration/test_serv04.py)). Eles pulam
sozinhos se `NOTEBOOKGPT_API_KEY` não estiver no ambiente.

Para ver a interface no navegador, sem abrir a janela nativa:

```bash
uv run flet run --web --port 8550 packages/notebookgpt-ui/src/notebookgpt_ui/app.py
```

No modo web o caderno é gravado em `SharedPreferences`, não em disco, porque o
navegador não expõe um diretório de documentos ao aplicativo.
