"""Ponto de entrada do NotebookGPT."""

from pathlib import Path

import flet as ft
from dotenv import load_dotenv
from notebookgpt_core import NotebookData, load_notebook, notebook_from_text, notebook_to_text, save_notebook
from notebookgpt_index import Indexer, VectorIndex
from notebookgpt_inference import InferenceSettings, LocalInferenceClient

from notebookgpt_ui.workspace import NotebookWorkspace

NOTEBOOK_FILENAME = "notebookgpt.txt"
INDEX_FILENAME = "index.db"
APP_DIRNAME = "notebookgpt"


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
        # Sem disco, o indice vive em memoria e e reconstruido a cada sessao.
        index_path = None

        async def persist(notebook: NotebookData) -> None:
            await preferences.set(NOTEBOOK_FILENAME, notebook_to_text(notebook))
    else:
        # Suporte, não documentos: o caderno é estado interno do aplicativo. A
        # pasta Documentos do Windows pode estar redirecionada para o OneDrive,
        # onde a Proteção contra Ransomware barra a escrita — era o que derrubava
        # a conversa no fim do streaming.
        #
        # A subpasta importa porque, rodando do código-fonte, o diretório de
        # suporte é o do runner do Flet (`%APPDATA%\Appveyor Systems Inc\Flet`,
        # medido), compartilhado com qualquer outro app Flet em desenvolvimento;
        # `org`/`company` do pyproject só valem em app empacotado.
        support_dir = await storage_paths.get_application_support_directory()
        notebook_path = Path(support_dir) / APP_DIRNAME / NOTEBOOK_FILENAME
        index_path = notebook_path.parent / INDEX_FILENAME
        initial_data = load_notebook(notebook_path)

        async def persist(notebook: NotebookData) -> None:
            save_notebook(notebook_path, notebook)

    client = LocalInferenceClient(InferenceSettings.from_env())
    # O indice e derivado: apagar o arquivo custa uma reindexacao, nao o caderno.
    indexer = Indexer(VectorIndex(index_path), client)
    page.render(NotebookWorkspace, page, picker, client, indexer, persist, initial_data)


def run():
    """Console script `notebookgpt`."""
    load_dotenv()
    ft.run(main)


if __name__ == "__main__":
    run()
