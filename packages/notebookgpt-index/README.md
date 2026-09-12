# notebookgpt-index

O banco vetorial embutido do [NotebookGPT](../../README.md).

Indexa as fontes TXT em uma tabela `vec0` do `sqlite-vec` e devolve os trechos mais
próximos da pergunta. Depende de `core` (pedaços e modelos) e de `inference` (os
embeddings vêm do serv04); nada aqui conhece a interface.
