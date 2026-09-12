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
