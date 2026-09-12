# notebookgpt-ui

A interface Flet do [NotebookGPT](../../README.md) — o pacote original deste
repositório. Três painéis: fontes à esquerda, conversa no meio, anotações à direita.

Escrita no estilo declarativo do Flet 0.86 (`@ft.component`, `ft.use_state`,
`page.render`). A resposta do modelo chega em streaming: cada trecho reemite a
lista de mensagens, e o caderno é gravado uma vez, no fim.

```bash
uv run notebookgpt
```
