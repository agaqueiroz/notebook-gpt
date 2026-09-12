"""Ponto de entrada do NotebookGPT."""

from pathlib import Path

import flet as ft
from dotenv import load_dotenv
from notebookgpt_core import NotebookData, load_notebook, notebook_from_text, notebook_to_text, save_notebook
from notebookgpt_inference import InferenceSettings, LocalInferenceClient

from notebookgpt_ui.workspace import NotebookWorkspace

NOTEBOOK_FILENAME = "notebookgpt.txt"


async def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(use_material3=True)
    page.padding = 0

    picker = ft.FilePicker()
    preferences = ft.SharedPreferences()
    storage_paths = ft.StoragePaths()
    page.services = [picker, preferences, storage_paths]

    # Decidir aqui onde o caderno mora deixa a interface sem saber nada sobre
    # armazenamento: ela recebe um `persist` e o chama.
    if page.web:
        # Navegadores não expõem um diretório de documentos ao aplicativo.
        # `page.platform` não serve para essa decisão: no navegador ele traz o
        # sistema operacional do visitante, não a ausência de plataforma.
        initial_data = notebook_from_text(await preferences.get(NOTEBOOK_FILENAME))

        async def persist(notebook: NotebookData) -> None:
            await preferences.set(NOTEBOOK_FILENAME, notebook_to_text(notebook))
    else:
        documents_dir = await storage_paths.get_application_documents_directory()
        notebook_path = Path(documents_dir) / NOTEBOOK_FILENAME
        initial_data = load_notebook(notebook_path)

        async def persist(notebook: NotebookData) -> None:
            save_notebook(notebook_path, notebook)

    client = LocalInferenceClient(InferenceSettings.from_env())
    page.render(NotebookWorkspace, page, picker, client, persist, initial_data)


def run():
    """Console script `notebookgpt`."""
    load_dotenv()
    ft.run(main)


if __name__ == "__main__":
    run()
