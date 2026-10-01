# -*- coding: utf-8 -*-
"""Carga de documentos DOCX y división en chunks.

Responsable de: DOCX loading → text extraction → RecursiveCharacterTextSplitter
→ metadata preservation (source, class, chunk_id).
"""

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR


def load_documents(data_dir: Path = None) -> list[Document]:
    """Carga los cuatro Knowledge Base del curso desde data/.

    Usa Docx2txtLoader, que extrae el texto de archivos DOCX y conserva
    'source' en los metadatos. Normaliza source al nombre de archivo
    (sin rutas locales con información del usuario).
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_dir}")

    loader = DirectoryLoader(
        str(data_dir),
        glob="*.docx",
        loader_cls=Docx2txtLoader,
    )
    documentos_crudos = loader.load()

    if not documentos_crudos:
        raise RuntimeError(
            "No se cargó ningún documento desde data/ — revisá que existan clase_1..4.docx"
        )

    # Normalizamos source al nombre de archivo (sin rutas locales)
    for doc in documentos_crudos:
        doc.metadata["source"] = Path(doc.metadata["source"]).name

    return documentos_crudos


def split_documents(documentos_crudos: list[Document]) -> list[Document]:
    """Divide los documentos en chunks con la configuración baseline del curso.

    El overlap evita cortar ideas a la mitad en los límites de fragmento.
    Cada chunk conserva metadatos separados del texto (source, class, chunk_id)
    — nunca JSON dentro de page_content.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documentos_crudos)

    if not chunks:
        raise RuntimeError("La colección de chunks está vacía — no hay nada que indexar")

    # Metadata avanzada: fuente, clase (derivada del nombre de archivo) e índice de chunk
    for i, chunk in enumerate(chunks):
        nombre = os.path.basename(chunk.metadata["source"])
        chunk.metadata["source"] = nombre
        # 'class' se deriva del nombre de archivo: clase_3.docx -> 3
        chunk.metadata["class"] = int(nombre.split("_")[1].split(".")[0])
        chunk.metadata["chunk_id"] = i

    return chunks


def load_and_split(data_dir: Path | None = None) -> list[Document]:
    """Punto de entrada: DOCX → chunks con metadatos preservados."""
    documentos = load_documents(data_dir)
    print(f"[OK] {len(documentos)} documentos cargados")
    chunks = split_documents(documentos)
    print(f"[OK] {len(chunks)} chunks generados ({CHUNK_SIZE}/{CHUNK_OVERLAP})")
    return chunks
