"""Workspace declarativo do NotebookGPT."""

from collections.abc import Awaitable, Callable

import flet as ft
from notebookgpt_core import (
    Chunk,
    Message,
    NotebookData,
    Source,
    StorageError,
    UnsupportedSourceError,
    build_messages,
    fits_whole,
    source_from_bytes,
)
from notebookgpt_index import Indexer, VectorIndexError
from notebookgpt_inference import InferenceError, LocalInferenceClient

Persist = Callable[[NotebookData], Awaitable[None]]


@ft.component
def NotebookWorkspace(
    page: ft.Page,
    picker: ft.FilePicker,
    client: LocalInferenceClient,
    indexer: Indexer,
    persist: Persist,
    initial_data: NotebookData,
):
    sources, set_sources = ft.use_state(initial_data.sources)
    messages, set_messages = ft.use_state(initial_data.messages)
    notes, set_notes = ft.use_state(initial_data.notes)
    prompt, set_prompt = ft.use_state("")
    sending, set_sending = ft.use_state(False)
    indexing, set_indexing = ft.use_state(False)
    left_visible, set_left_visible = ft.use_state(True)
    right_visible, set_right_visible = ft.use_state(True)

    # O ref guarda a versão gravável do caderno e sobrevive às renderizações.
    # Sem ele, cada handler fecharia sobre o estado da renderização em que foi
    # criado: uma anotação escrita durante o streaming, por exemplo, era
    # sobrescrita pelo `persist` do fim da resposta.
    # Divisão: os `use_state` desenham a tela, o ref é o que vai para o disco.
    stored = ft.use_ref(
        lambda: NotebookData(
            sources=initial_data.sources,
            messages=initial_data.messages,
            notes=initial_data.notes,
        )
    )

    # Uma falha de gravação avisa uma vez e cala. `change_notes` dispara a cada
    # tecla: com o disco barrado, seriam dezenas de SnackBar idênticos.
    storage_warned = ft.use_ref(False)
    # Idem para o indice: sem ele a conversa segue com o contexto truncado, e
    # repetir o aviso a cada pergunta so atrapalharia.
    index_warned = ft.use_ref(False)

    async def commit(
        next_sources: list[Source] | None = None,
        next_messages: list[Message] | None = None,
        next_notes: str | None = None,
    ) -> None:
        """Aplica a mudança no estado da tela e no caderno, e grava.

        A ordem importa: o estado da tela é atualizado antes da gravação, então
        uma resposta que acabou de chegar continua visível mesmo se o disco
        recusar a escrita. Perder a conversa por causa do arquivo não é opção.
        """
        notebook = stored.current
        if next_sources is not None:
            notebook.sources = next_sources
            set_sources(next_sources)
        if next_messages is not None:
            notebook.messages = next_messages
            set_messages(next_messages)
        if next_notes is not None:
            notebook.notes = next_notes
            set_notes(next_notes)

        try:
            await persist(notebook)
        except StorageError as error:
            if not storage_warned.current:
                storage_warned.current = True
                warn(str(error))

    def warn(text: str) -> None:
        page.show_dialog(ft.SnackBar(content=ft.Text(text)))

    async def add_files(_):
        selected = await picker.pick_files(
            dialog_title="Adicionar fontes (.txt)",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["txt"],
            allow_multiple=True,
            # Os bytes vêm no próprio evento: é o único caminho que também
            # funciona no modo web, onde `path` é sempre None.
            with_data=True,
        )
        current = stored.current.sources
        known = {source.name for source in current}
        added: list[Source] = []
        for file in selected:
            if file.name in known:
                warn(f"'{file.name}' já está na lista de fontes.")
                continue
            if file.bytes is None:
                warn(f"Não foi possível ler o conteúdo de '{file.name}'.")
                continue
            try:
                added.append(source_from_bytes(file.name, file.bytes))
                known.add(file.name)
            except UnsupportedSourceError as error:
                warn(str(error))

        if added:
            await commit(next_sources=current + added)
            await reindex()

    def remove_handler(name: str):
        async def remove_source(_):
            await commit(next_sources=[s for s in stored.current.sources if s.name != name])
            await reindex()

        return remove_source

    # Fontes anexadas enquanto uma resposta corre: o serv04 atende uma inferência
    # por vez e devolve 503 se receber duas. Fica pendente e indexa quando a
    # resposta terminar.
    pending_index = ft.use_ref(False)
    # O espelho de `sending` num ref, porque `reindex` precisa do valor de agora e
    # não do valor da renderização em que foi criada: ao drenar a fila logo após
    # `set_sending(False)`, a variável de estado ainda diz True e o trabalho ficaria
    # adiado para sempre.
    streaming = ft.use_ref(False)

    async def reindex() -> None:
        """Poe o banco vetorial em dia com as fontes do caderno.

        Acontece ao anexar ou remover, que é quando o usuário já espera trabalho —
        e não no meio de uma pergunta. Falhar aqui não impede nada: sem índice, a
        conversa volta ao contexto truncado.
        """
        if streaming.current:
            pending_index.current = True
            return

        pending_index.current = False
        set_indexing(True)
        try:
            await indexer.sync(stored.current.sources)
        except (InferenceError, VectorIndexError) as error:
            if not index_warned.current:
                index_warned.current = True
                warn(f"Índice indisponível, seguindo sem busca por similaridade. {error}")
        finally:
            set_indexing(False)

    # Na abertura, o caderno pode ter fontes que o índice não tem: no modo web ele
    # vive em memória e nasce vazio a cada sessão, e no desktop o `index.db` pode
    # ter sido apagado. Sem isto, a primeira pergunta iria ao modelo dizendo que
    # não há fontes. `sync` é incremental, então quando o índice já está em dia
    # isto custa só a comparação dos digests.
    async def sync_on_open() -> None:
        if initial_data.sources:
            await reindex()

    ft.use_effect(sync_on_open, [])

    async def gather_passages(question: str) -> list[Chunk] | None:
        """Os trechos que respondem à pergunta, ou None para mandar tudo.

        Cabendo as fontes inteiras no orçamento, recuperar só subtrai: os trechos
        descartados teriam cabido de qualquer jeito.
        """
        if fits_whole(stored.current.sources):
            return None
        try:
            passages = await indexer.retrieve(question)
        except (InferenceError, VectorIndexError) as error:
            if not index_warned.current:
                index_warned.current = True
                warn(f"Busca por similaridade falhou, usando o contexto truncado. {error}")
            return None

        # Índice vazio com fontes na mão: melhor mandar o contexto truncado do que
        # dizer ao modelo que não há fonte nenhuma.
        return passages or None

    async def submit_prompt(_):
        clean_prompt = prompt.strip()
        if not clean_prompt or sending:
            return

        notebook = stored.current
        history = [*notebook.messages, Message(role="user", content=clean_prompt)]

        set_prompt("")
        streaming.current = True
        set_sending(True)
        set_messages([*history, Message(role="assistant", content="")])

        passages = await gather_passages(clean_prompt)
        request = build_messages(notebook, clean_prompt, passages)
        used = passages or []

        answer = ""
        try:
            async for delta in client.stream_reply(request):
                answer += delta
                set_messages([*history, Message(role="assistant", content=answer, passages=used)])
        except InferenceError as error:
            # O erro entra na conversa: é onde o usuário está olhando, e fica
            # gravado junto do prompt que o causou.
            answer = f"**Falha na conversa com o servidor.**\n\n{error}"
            warn(str(error))

        if not answer:
            answer = "_O servidor respondeu sem conteúdo._"

        streaming.current = False
        set_sending(False)
        await commit(next_messages=[*history, Message(role="assistant", content=answer, passages=used)])

        if pending_index.current:
            # `streaming` ja e False aqui, entao `reindex` vai em frente.
            await reindex()

    def change_prompt(event):
        set_prompt(event.control.value)

    async def change_notes(event):
        await commit(next_notes=event.control.value)

    def copy_handler(markdown: str):
        async def copy_answer(_):
            await page.clipboard.set(markdown)

        return copy_answer

    def conversation_item(message: Message) -> ft.Control:
        if message.role == "user":
            return ft.Container(
                content=ft.Column(controls=[ft.Text("Você"), ft.Text(message.content, selectable=True)]),
                padding=12,
            )

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("NotebookGPT"),
                    ft.Markdown(
                        value=message.content or "…",
                        selectable=True,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
                    ),
                    ft.Row(
                        controls=[
                            ft.Button(
                                content="Copiar",
                                icon=ft.Icons.CONTENT_COPY,
                                on_click=copy_handler(message.content),
                            )
                        ],
                        tight=True,
                    ),
                    *passages_panel(message.passages),
                ]
            ),
            padding=12,
        )

    def passages_panel(passages: list[Chunk]) -> list[ft.Control]:
        """Os trechos que embasaram a resposta, recolhidos.

        Num PoC de RAG, poder abrir e ver o que o modelo recebeu é metade do valor:
        quando a resposta sai errada, é isto que diz se a culpa foi da busca ou do
        modelo.
        """
        if not passages:
            return []

        return [
            ft.ExpansionTile(
                title=ft.Text(f"{len(passages)} trecho(s) consultado(s)", size=12),
                dense=True,
                controls=[
                    ft.Container(
                        padding=ft.Padding.only(left=12, right=12, bottom=8),
                        content=ft.Column(
                            spacing=2,
                            controls=[
                                ft.Text(
                                    f"{passage.source_name} · trecho {passage.ordinal} de {passage.total}",
                                    size=11,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.Text(passage.content, size=11, selectable=True),
                            ],
                        ),
                    )
                    for passage in passages
                ],
            )
        ]

    def source_item(source: Source) -> ft.Control:
        return ft.Row(
            controls=[
                ft.Column(
                    expand=True,
                    spacing=0,
                    controls=[
                        ft.Text(source.name, no_wrap=True),
                        ft.Text(f"{len(source.content)} caracteres", size=11),
                    ],
                ),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_size=16,
                    tooltip="Remover fonte",
                    on_click=remove_handler(source.name),
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    source_panel = ft.Container(
        width=280,
        padding=12,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        content=ft.Column(
            controls=[
                ft.Button(
                    content="Adicionar arquivos",
                    icon=ft.Icons.FILE_UPLOAD,
                    on_click=add_files,
                ),
                ft.Text("Somente .txt nesta versão.", size=11),
                ft.Row(
                    controls=[ft.ProgressRing(width=12, height=12), ft.Text("Indexando…", size=11)],
                    tight=True,
                )
                if indexing
                else ft.Text(
                    "Busca por similaridade ativa." if not fits_whole(sources) else "Fontes cabem inteiras; sem busca.",
                    size=11,
                ),
                ft.Column(controls=[source_item(source) for source in sources]),
            ]
        ),
    )

    conversation = ft.Column(
        expand=True,
        controls=[
            ft.ListView(
                expand=True,
                auto_scroll=True,
                controls=(
                    [ft.Text("Envie uma mensagem para iniciar a conversa.")]
                    if not messages
                    else [conversation_item(message) for message in messages]
                ),
            ),
            ft.Row(
                controls=[
                    ft.TextField(
                        value=prompt,
                        hint_text="Pergunte sobre suas fontes",
                        multiline=True,
                        min_lines=1,
                        max_lines=4,
                        shift_enter=True,
                        expand=True,
                        disabled=sending,
                        on_change=change_prompt,
                        on_submit=submit_prompt,
                    ),
                    ft.Container(
                        width=40,
                        height=40,
                        alignment=ft.Alignment.CENTER,
                        content=(
                            ft.ProgressRing(width=18, height=18)
                            if sending
                            else ft.IconButton(
                                icon=ft.Icons.SEND,
                                on_click=submit_prompt,
                                tooltip="Enviar",
                            )
                        ),
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.END,
            ),
        ],
    )

    notes_panel = ft.Container(
        width=300,
        padding=12,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        content=ft.Column(
            controls=[
                ft.Text("Anotações"),
                ft.TextField(
                    value=notes,
                    hint_text="Escreva suas anotações aqui",
                    multiline=True,
                    min_lines=8,
                    max_lines=20,
                    expand=True,
                    on_change=change_notes,
                ),
            ],
            expand=True,
        ),
    )

    return ft.Column(
        expand=True,
        controls=[
            ft.AppBar(
                bgcolor=ft.Colors.SURFACE_CONTAINER,
                leading=ft.IconButton(
                    icon=ft.Icons.MENU,
                    on_click=lambda _: set_left_visible(not left_visible),
                    tooltip="Mostrar ou ocultar fontes",
                ),
                title=ft.Text(f"NotebookGPT · {client.settings.model} em {client.settings.host}"),
                center_title=True,
                actions=[
                    ft.IconButton(
                        icon=ft.Icons.SETTINGS,
                        on_click=lambda _: set_right_visible(not right_visible),
                        tooltip="Mostrar ou ocultar anotações",
                    )
                ],
            ),
            ft.Row(
                expand=True,
                controls=(
                    ([source_panel] if left_visible else []) + [conversation] + ([notes_panel] if right_visible else [])
                ),
            ),
        ],
    )
