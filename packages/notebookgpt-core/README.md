# notebookgpt-core

O domínio do [NotebookGPT](../../README.md): o caderno (`NotebookData`, `Message`,
`Source`), a persistência em TXT, a leitura das fontes e a montagem do contexto
enviado ao modelo.

Só depende da biblioteca padrão. Nada aqui conhece Flet ou HTTP — é o que permite
testar as regras do PoC sem janela e sem rede.
