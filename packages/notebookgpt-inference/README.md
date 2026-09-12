# notebookgpt-inference

A camada anticorrupção entre o [NotebookGPT](../../README.md) e o servidor local de
inferência ([local-inference-server](https://github.com/agaqueiroz/local-inference-server)).

Como o servidor implementa a API da OpenAI, o cliente é o SDK oficial — e é
justamente isso que prova a compatibilidade. Para fora, este pacote expõe apenas
`LocalInferenceClient.stream_reply()`, que emite trechos de texto, e
`InferenceError`, com mensagens prontas para a interface. Nada acima desta camada
importa `openai`.
