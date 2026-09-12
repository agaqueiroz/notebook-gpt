# TODO — NotebookGPT

O PoC já conversa com o `serv04.local` em streaming, usando as fontes TXT como
contexto. O que falta é o que ficou deliberadamente de fora dele.

## Contexto e fontes

- [ ] Aceitar outros formatos além de TXT (PDF, DOCX, MD), extraindo texto e metadados
      antes de enviar ao modelo. O portão está em `notebookgpt_core.sources.SUPPORTED_SUFFIXES`.
- [x] ~~Trocar o contexto por RAG usando `POST /v1/embeddings` do próprio serv04.~~
      Feito: banco vetorial embutido com `sqlite-vec`, híbrido (fontes inteiras quando
      cabem, recuperação quando não).
- [ ] Contar tokens em vez de caracteres, para usar o `LIS_MAX_MODEL_LENGTH` real
      em vez das constantes conservadoras de `context.py` e `chunking.py`.

## Prefixo do e5 — pendência de estudo

- [ ] **Esvaziar `LIS_EMBEDDING_INPUT_PREFIX` no serv04.** Medido pelo contador de
      tokens, o servidor ainda prepende `query: ` (3 tokens) a toda entrada, então
      os prefixos do cliente estão sendo dobrados: o modelo vê `query: passage: …`
      ao indexar. Enquanto não for feito, dá para desligar o lado do cliente com
      `NOTEBOOKGPT_PASSAGE_PREFIX=` e `NOTEBOOKGPT_QUERY_PREFIX=` no `.env`.
- [ ] Entender **quanto isso custa de fato**. O recall medido foi 12/12 nas três
      configurações (correta, dobrada e sem prefixo nenhum), ou seja, neste corpus a
      escolha não aparece. Falta um caso em que apareça — provavelmente corpus maior,
      perguntas mais ambíguas ou trechos mais parecidos entre si. Sem esse caso, a
      recomendação do e5 é fé na documentação, não medição.
- [ ] Propor ao `local-inference-server` um prefixo **por requisição** (ou um campo
      que distinga documento de pergunta). A API da OpenAI não tem esse conceito, mas
      qualquer cliente de RAG contra esse servidor esbarra no mesmo limite.

## Banco vetorial

- [ ] Trechos com fronteira melhor: um trecho pode começar com pontuação órfã vinda
      da sobreposição. É cosmético para o embedding, mas suja o painel de trechos.
- [ ] Reindexação em segundo plano para arquivos grandes: hoje anexar um TXT longo
      segura a interface até os embeddings voltarem.
- [ ] `index.db` cresce com o histórico de fontes removidas? Não — `drop_source`
      limpa —, mas falta um `VACUUM` ocasional para o arquivo encolher de fato.

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

## Limites do modelo de 1B

- [ ] **Sensibilidade de frase.** "Quantas telhas o granizo perfurou?" recusa 3 de 3
      execucoes com o trecho certo no contexto e "18 telhas" visivel; "O granizo
      perfurou quantas telhas de policarbonato?" responde certo. Nao e variancia nem
      falha de busca. Um reescritor de pergunta resolveria, mas custa uma inferencia
      a mais por pergunta -- medir se compensa nesta GPU antes de adotar.
- [ ] **Transcricao numerica.** Numeros escritos por extenso na fonte ("quatrocentos
      e trinta") voltam errados na resposta ("4 pedidos"). Com algarismos no texto o
      acerto e bom. Se as suas fontes escreverem numeros por extenso, isto aparece.
- [ ] Avaliar o Gemma-3-Gaia-PT-BR-4b que o servidor suporta. Fica fora do escopo
      atual por custar mais GPU, mas e o caminho obvio se a qualidade do 1B travar.

## Concorrencia no servidor

- [x] ~~O serv04 atende uma inferencia por vez e devolve 503 `server_busy`.~~ O
      cliente traduz para um aviso legivel e a interface adia a indexacao enquanto
      uma resposta esta em streaming.
- [ ] A indexacao de um arquivo grande ainda segura a interface ate os embeddings
      voltarem; trabalho em segundo plano com barra de progresso resolveria.
