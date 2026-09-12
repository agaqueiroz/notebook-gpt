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
