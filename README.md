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
uv run notebookgpt
```

Antes da primeira execução, crie um `.env` na raiz com a chave do servidor — é a
mesma `LIS_API_KEY` configurada lá. O `.env` não é versionado, e não há gabarito
rastreado no repositório justamente para não haver onde um segredo caia por
descuido; a tabela abaixo é a referência.

```bash
cat > .env <<'EOF'
NOTEBOOKGPT_BASE_URL=http://serv04.local:8000/v1
NOTEBOOKGPT_API_KEY=cole-a-sua-chave-aqui
NOTEBOOKGPT_MODEL=llama-3.2-1b
EOF
```

| Variável | Padrão | O que é |
| --- | --- | --- |
| `NOTEBOOKGPT_BASE_URL` | `http://serv04.local:8000/v1` | Endereço do servidor |
| `NOTEBOOKGPT_API_KEY` | *(vazio)* | Chave exigida pelo servidor |
| `NOTEBOOKGPT_MODEL` | `llama-3.2-1b` | Modelo de chat |
| `NOTEBOOKGPT_EMBEDDING_MODEL` | `multilingual-e5-small` | Modelo de embeddings, para a busca |
| `NOTEBOOKGPT_PASSAGE_PREFIX` | `passage: ` | Prefixo do e5 ao indexar (ver abaixo) |
| `NOTEBOOKGPT_QUERY_PREFIX` | `query: ` | Prefixo do e5 ao perguntar (ver abaixo) |
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
| `notebookgpt-core` | Caderno, fontes TXT, persistência, pedaços e montagem do contexto | biblioteca padrão |
| `notebookgpt-inference` | Cliente do servidor local: streaming, embeddings e tradução de erros | `core`, `openai` |
| `notebookgpt-index` | Banco vetorial embutido e recuperação dos trechos | `core`, `inference`, `sqlite-vec` |
| `notebookgpt-ui` | Interface Flet (o pacote original deste repositório) | os três acima, `flet` |

A direção é sempre `ui → index → inference → core`. O domínio não conhece rede nem
interface, e o cliente de inferência não conhece a interface — é o que permite
testar a montagem do contexto e o streaming sem abrir uma janela.
[tests/test_layers.py](tests/test_layers.py) verifica isso no código-fonte.

## Fontes de contexto

Nesta etapa, **somente `.txt`**. O seletor de arquivos já filtra por extensão e
`notebookgpt_core.sources` recusa o resto com `UnsupportedSourceError`.

As fontes são lidas no momento em que você as adiciona e ficam gravadas junto do
caderno — é o que faz elas sobreviverem a um reinício também no modo web, onde o
Flet nunca devolve o caminho do arquivo.

## O banco vetorial e o RAG

O servidor roda com `LIS_MAX_MODEL_LENGTH=4096`, então nem todo TXT cabe no
contexto. Em vez de cortar e torcer, o NotebookGPT indexa as fontes num **banco
vetorial embutido** e manda ao modelo só os trechos que respondem à pergunta.

O caminho é híbrido, e essa é a decisão que mais importa:

- **cabendo as fontes inteiras** no orçamento de 6000 caracteres, elas vão inteiras.
  Recuperar num corpus pequeno só subtrai — os trechos descartados teriam cabido.
- **acima disso**, entra a busca por similaridade.

O banco é um arquivo SQLite com a extensão [`sqlite-vec`](https://github.com/asg017/sqlite-vec):
uma tabela virtual `vec0` faz o KNN em SQL, e uma coluna auxiliar `+content` guarda
o texto do trecho, então a busca já devolve o conteúdo. Fica em `index.db`, ao lado
do caderno; no modo web vive em memória e é reconstruído a cada sessão. É **derivado**:
apagar o arquivo custa uma reindexação, nunca o caderno.

A reindexação é incremental — o `sha256` do conteúdo de cada fonte fica gravado, e
reabrir o aplicativo não gasta uma rodada de embeddings.

### O que foi medido

| Fato | Valor | O que decidiu |
| --- | --- | --- |
| Dimensões do `multilingual-e5-small` | 384 | `float[384]` fixo; `dimensions` é recusado com 400 |
| Norma dos vetores devolvidos | **1.0** | cosseno **é** produto escalar, e a distância L2 padrão do `vec0` ordena igual — sem NumPy, sem métrica declarada |
| Entradas por requisição | 64 (erro 400 em 65) | o lote de `LocalInferenceClient.embed` |
| Teto de tokens do embedding | 511, truncagem silenciosa | o tamanho do trecho |
| Caracteres por token (pt-BR) | ~3,9 | 511 tokens ≈ 2000 chars, daí `CHUNK_CHARS = 1200` |
| Similaridade entre textos sem relação | **0,81** | **não existe limiar útil**: o corte é por ranking e orçamento |
| Recall@1 em 12 perguntas com distratores próximos | **12/12** | o piso de 9 em `tests/integration/test_rag.py` |

### O prefixo do e5 — pendência no servidor

O e5 foi treinado com `query: ` para a pergunta e `passage: ` para o documento. O
servidor aplica **um único** prefixo a toda entrada (`LIS_EMBEDDING_INPUT_PREFIX`),
sem o cliente poder escolher, então os dois prefixos corretos só funcionam com ele
esvaziado no serv04.

Medindo o contador de tokens (`tokens(X)` menos o custo marginal de uma cópia de
`X` revela o overhead fixo), **o servidor ainda prepende `query: `** — 3 tokens,
batendo exatamente com `t("query: ")`. Enquanto for assim, o cliente está dobrando
o prefixo. Na prática isso **não mudou nada** no recall medido (12/12 com prefixo,
sem prefixo e com prefixo dobrado), mas é errado e está anotado em
[TODO.md](TODO.md). Para desligar o lado do cliente até lá, basta
`NOTEBOOKGPT_PASSAGE_PREFIX=` e `NOTEBOOKGPT_QUERY_PREFIX=` no `.env`.

### O formato do contexto não é cosmético

Duas decisões de `notebookgpt_core.context` foram **medidas** contra o
`llama-3.2-1b` do serv04, com `temperature=0`, em cinco perguntas cuja resposta
está na fonte anexada:

| Formato das fontes | Onde elas entram | Respostas fundamentadas |
| --- | --- | --- |
| Cabeçalhos Markdown (`## Fontes anexadas`) | `system` | 1/5 |
| Cabeçalhos Markdown | turno do usuário | 0/5 |
| Texto delimitado (`Arquivo "x.txt": """…"""`) | `system` | 0/5 |
| Texto delimitado | turno do usuário | **4/5** |

Nenhuma das duas mudanças resolve sozinha — só a combinação. Um modelo pequeno
ignora o que está em `system` e trata o cabeçalho Markdown como conteúdo a
interpretar, não como moldura; nas três primeiras linhas ele responde coisas como
"não tenho acesso a essa informação" ou "não posso responder a esta pergunta",
às vezes citando o arquivo na própria recusa.

Mesmo com `temperature=0` o servidor não é perfeitamente determinístico (a linha
de 0/5 já apareceu como 2/5 em outra rodada), e é por isso que
[tests/integration/test_serv04.py](tests/integration/test_serv04.py) fixa um piso
de 3/5 em vez de um número exato: chega para derrubar uma regressão de formato
sem quebrar a cada execução.

### A armadilha do system prompt

A primeira versão do `SYSTEM_PROMPT` trazia uma salvaguarda que parecia óbvia:
*"se elas não contiverem a resposta, diga isso claramente em vez de inventar"*.
Ela causava o bug oposto — o modelo recusando perguntas cuja resposta estava no
contexto, às vezes **enunciando o fato dentro da negativa**:

> "Não há informações sobre o portao norte ficando inoperante por **9 dias**,
> conforme mencionado na **Ocorrência 3**"

Medido sobre oito perguntas de detalhe, num corpus onde a busca acertou 8/8 e
sempre em primeiro lugar:

| System prompt | Acertos | Recusas |
| --- | --- | --- |
| com a licença de recusa | 3/8 | 5/8 |
| **sem qualquer menção a ausência** | **6/8** | **0/8** |
| "responda direto; só declare ausência depois de reler" | 0/8 | **8/8** |

A terceira linha é a lição, e é contraintuitiva: **a menção contamina mesmo quando
a instrução é para não usá-la.** Um modelo pequeno trata "não encontrar" como
molde disponível, não como condição a avaliar, e uma pergunta em forma de consulta
factual basta para disparar o molde antes de ele ler o texto.

O medo que motivava a cláusula era infundado: em perguntas cujo fato não existe no
corpus, o modelo **declinou em todas**. A capacidade de recusar continua lá; ela só
parou de disparar à toa. [tests/integration/test_recusas.py](tests/integration/test_recusas.py)
fixa as duas metades, porque uma sem a outra seria um retrocesso.

O que **não** ajudou, também medido: ampliar o orçamento de 6000 para 10000
caracteres (6/8 nos dois) e incluir os trechos vizinhos ao que respondeu, que
piorou — misturou duas ocorrências numa resposta e errou uma contagem. Num modelo
de 1B, contexto extra é distrator.

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
navegador não expõe um diretório de arquivos ao aplicativo.

## Uma inferência por vez

O serv04 atende um pedido por vez e devolve `503 server_busy` depois de
`LIS_BUSY_TIMEOUT` (30 s). O NotebookGPT traduz isso para um aviso que diz o que
está acontecendo, em vez de "erro 503", e não indexa enquanto uma resposta está em
streaming — anexar um arquivo nesse momento deixa a indexação pendente e ela
acontece assim que a resposta termina.

## Onde o caderno é gravado, e por que ali

No desktop, a conversa e as anotações vão para uma subpasta `notebookgpt/` dentro
do **diretório de suporte do aplicativo** (`get_application_support_directory()`),
não para a pasta Documentos. Onde isso cai depende de como o app foi iniciado:

| Como roda | Caminho medido |
| --- | --- |
| `uv run notebookgpt` (desenvolvimento) | `%APPDATA%\Appveyor Systems Inc\Flet\notebookgpt\` |
| Empacotado por `flet build` | `%APPDATA%\<org>\<app>\notebookgpt\` |

Em desenvolvimento o diretório é o do próprio runner do Flet, compartilhado com
qualquer outro app Flet rodando do código-fonte — o `org` e o `company` do
`pyproject.toml` só valem no app empacotado. Daí a subpasta: ela garante que o
caderno tenha um canto próprio nos dois casos.

Isso não é preferência de arrumação. A pasta Documentos do Windows pode estar
redirecionada para o OneDrive, e a **Proteção contra Ransomware** (Acesso
Controlado a Pastas) bloqueia a escrita ali. O detalhe cruel é que o bloqueio
chega ao Python como `[Errno 2] No such file or directory` em vez de permissão
negada — quem lê a mensagem vai procurar um arquivo que falta, não uma permissão.
Foi o que derrubava a conversa no fim do streaming.

As fontes de contexto são **somente leitura**: o NotebookGPT lê o `.txt` que você
anexa e nunca escreve nele nem o move. `tests/core/test_sources.py` fixa isso
comparando conteúdo e `mtime` antes e depois da leitura.

E se a gravação falhar de todo modo — disco cheio, antivírus, permissão — a
conversa continua na tela e um aviso aparece uma vez. O caderno é o registro, não
a conversa.
