# TODO — NotebookGPT

O PoC já conversa com o `serv04.local` em streaming, usando as fontes TXT como
contexto. O que falta é o que ficou deliberadamente de fora dele.

## Contexto e fontes

- [ ] Aceitar outros formatos além de TXT (PDF, DOCX, MD), extraindo texto e metadados
      antes de enviar ao modelo. O portão está em `notebookgpt_core.sources.SUPPORTED_SUFFIXES`.
- [ ] Trocar o contexto por RAG usando `POST /v1/embeddings` do próprio serv04
      (`multilingual-e5-small` já está carregado lá). Hoje `notebookgpt_core.context`
      corta as fontes por orçamento de caracteres, o que não escala para arquivos grandes.
- [ ] Contar tokens em vez de caracteres, para usar o `LIS_MAX_MODEL_LENGTH` real
      em vez das constantes conservadoras de `context.py`.

## Conversa

- [ ] Permitir interromper uma resposta em andamento (hoje o streaming vai até o fim).
- [ ] Reaproveitar `POST /v1/models` para oferecer a escolha do modelo na interface,
      em vez de só `NOTEBOOKGPT_MODEL`.

## Persistência

- [ ] Em desktop e mobile, o estado é salvo em `notebookgpt.txt`, no diretório de
      documentos do aplicativo; no navegador, em `SharedPreferences`.
- [ ] O conteúdo das fontes é gravado junto do caderno — é o que faz elas sobreviverem
      a um reinício no modo web. Migrar para SQLite quando houver múltiplos cadernos,
      busca ou anexos grandes.

## Empacotamento

- [ ] Ligar `flet build` ao layout do monorepo. Hoje ele não funciona: espera o
      módulo de entrada na raiz do diretório do app (`tool.flet.app.path` +
      `tool.flet.app.module`, que resolvem para um arquivo, não para um pacote
      `src/`) e instala só as dependências do próprio `pyproject.toml`, o que não
      alcança `notebookgpt-core` nem `notebookgpt-inference`. O caminho suportado
      é `uv run notebookgpt`.

## Persistência, pontas soltas

- [ ] Migrar um `notebookgpt.txt` gravado por versões antigas na pasta Documentos.
      Nesta máquina o arquivo nunca chegou a existir (a escrita era bloqueada), então
      não há o que mover e a migração seria código morto — mas quem rodou uma versão
      antiga em outra máquina tem um caderno preso lá. A leitura daquela pasta não é
      bloqueada; só a escrita.
- [ ] Distinguir "o caderno não existe" de "existe e não abre" em `load_notebook`.
      Hoje os dois casos devolvem um caderno vazio em silêncio, e foi esse silêncio
      que escondeu o bug da pasta protegida por um tempo.
