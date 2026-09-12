"""Ponto de entrada do NotebookGPT."""

from pathlib import Path

import flet as ft
from dotenv import load_dotenv
from notebookgpt_core import load_notebook, notebook_from_text
from notebookgpt_inference import InferenceSettings, LocalInferenceClient

from notebookgpt_ui.workspace import NotebookWorkspace

NOTEBOOK_FILENAME = "notebookgpt.txt"


async def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(use_material3=True)
    page.padding = 0

    if page.platform is None:
        # Navegadores não expõem um diretório de documentos ao aplicativo.
        storage_path = None
        saved_text = await page.shared_preferences.get(NOTEBOOK_FILENAME)
        initial_data = notebook_from_text(saved_text)
    else:
        documents_dir = await page.storage_paths.get_application_documents_directory()
        storage_path = Path(documents_dir) / NOTEBOOK_FILENAME
        initial_data = load_notebook(storage_path)

    picker = ft.FilePicker()
    client = LocalInferenceClient(InferenceSettings.from_env())
    page.render(NotebookWorkspace, page, picker, client, storage_path, initial_data)


def run():
    """Console script `notebookgpt`."""
    load_dotenv()
    ft.run(main)


if __name__ == "__main__":
    run()
