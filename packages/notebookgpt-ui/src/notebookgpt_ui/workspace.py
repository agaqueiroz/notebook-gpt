"""Workspace declarativo do NotebookGPT."""

from pathlib import Path

import flet as ft
from notebookgpt_core import (
    Message,
    NotebookData,
    Source,
    UnsupportedSourceError,
    build_messages,
    notebook_to_text,
    save_notebook,
    source_from_bytes,
)
from notebookgpt_inference import InferenceError, LocalInferenceClient

NOTEBOOK_FILENAME = "notebookgpt.txt"


@ft.component
def NotebookWorkspace(
    page: ft.Page,
    picker: ft.FilePicker,
    client: LocalInferenceClient,
    storage_path: Path | None,
    initial_data: NotebookData,
):
    sources, set_sources = ft.use_state(initial_data.sources)
    messages, set_messages = ft.use_state(initial_data.messages)
    notes, set_notes = ft.use_state(initial_data.notes)
    prompt, set_prompt = ft.use_state("")
    sending, set_sending = ft.use_state(False)
    left_visible, set_left_visible = ft.use_state(True)
    right_visible, set_right_visible = ft.use_state(True)

    async def save_state(
        next_sources: list[Source] = sources,
        next_messages: list[Message] = messages,
        next_notes: str = notes,
    ) -> None:
        notebook = NotebookData(
            sources=next_sources,
            messages=next_messages,
            notes=next_notes,
        )
        if storage_path is not None:
            save_notebook(storage_path, notebook)
        else:
            await page.shared_preferences.set(NOTEBOOK_FILENAME, notebook_to_text(notebook))

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
        known = {source.name for source in sources}
        added: list[Source] = []
        for file in selected:
            if file.name in known or file.bytes is None:
                continue
            try:
                added.append(source_from_bytes(file.name, file.bytes))
                known.add(file.name)
            except UnsupportedSourceError as error:
                warn(str(error))

        if not added:
            return

        next_sources = sources + added
        set_sources(next_sources)
        await save_state(next_sources=next_sources)

    def remove_handler(name: str):
        async def remove_source(_):
            next_sources = [source for source in sources if source.name != name]
            set_sources(next_sources)
            await save_state(next_sources=next_sources)

        return remove_source

    async def submit_prompt(_):
        clean_prompt = prompt.strip()
        if not clean_prompt or sending:
            return

        history = [*messages, Message(role="user", content=clean_prompt)]
        request = build_messages(NotebookData(sources=sources, messages=messages), clean_prompt)

        set_prompt("")
        set_sending(True)
        set_messages([*history, Message(role="assistant", content="")])

        answer = ""
        try:
            async for delta in client.stream_reply(request):
                answer += delta
                set_messages([*history, Message(role="assistant", content=answer)])
        except InferenceError as error:
            # O erro entra na conversa: é onde o usuário está olhando, e fica
            # gravado junto do prompt que o causou.
            answer = f"**Falha na conversa com o servidor.**\n\n{error}"
            warn(str(error))

        if not answer:
            answer = "_O servidor respondeu sem conteúdo._"

        next_messages = [*history, Message(role="assistant", content=answer)]
        set_messages(next_messages)
        set_sending(False)
        await save_state(next_messages=next_messages)

    def change_prompt(event):
        set_prompt(event.control.value)

    async def change_notes(event):
        next_notes = event.control.value
        set_notes(next_notes)
        await save_state(next_notes=next_notes)

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
                ]
            ),
            padding=12,
        )

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
