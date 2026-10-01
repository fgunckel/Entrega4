# -*- coding: utf-8 -*-
"""Pipeline de ingesta: chunks → HuggingFace embeddings → Pinecone Serverless.

Mantiene: all-MiniLM-L6-v2, 384 dimensiones, similitud coseno, AWS us-east-1.
El índice se crea solo si no existe. Los IDs de los vectores son deterministas
(fuente + chunk_id), así re-correr la ingesta actualiza en lugar de duplicar.
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Carga el modelo de embeddings local de HuggingFace.

    Es el mismo modelo para ingesta y consulta: si los embeddings del corpus
    y de la consulta vinieran de modelos distintos, la comparación de
    similitud no tendría sentido.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Verificamos la dimensión real del modelo contra la esperada
    vector_prueba = embeddings.embed_query("prueba de dimensión")
    dim_real = len(vector_prueba)
    if dim_real != EMBEDDING_DIM:
        raise RuntimeError(
            f"Mismatch de dimensión: el modelo produce {dim_real} dimensiones "
            f"pero EMBEDDING_DIM vale {EMBEDDING_DIM}."
        )

    print(f"[OK] Embedding model cargado: {EMBEDDING_MODEL} (dimensión {EMBEDDING_DIM})")
    return embeddings


def get_pinecone_index(api_key: str):
    """Conecta a Pinecone y crea el índice Serverless solo si no existe.

    Antes de crear verifica si ya existe para no recrearlo en cada ejecución.
    Luego valida que la dimensión y la métrica del índice coincidan con el modelo.
    """
    if not api_key:
        raise RuntimeError(
            "Falta PINECONE_API_KEY. Definila en el archivo .env "
            "(gratis en https://app.pinecone.io)."
        )

    try:
        pc = Pinecone(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"No se pudo conectar a Pinecone: {exc}") from exc

    indices_existentes = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in indices_existentes:
        print(f"[..] Creando índice '{PINECONE_INDEX_NAME}' (dimensión {EMBEDDING_DIM})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        print(f"[OK] El índice '{PINECONE_INDEX_NAME}' ya existe — se reutiliza")

    indice = pc.Index(PINECONE_INDEX_NAME)

    # Verificación de configuración: dimensión y métrica deben coincidir con el modelo
    config_indice = pc.describe_index(PINECONE_INDEX_NAME)
    if config_indice.dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"El índice '{PINECONE_INDEX_NAME}' tiene dimensión {config_indice.dimension} "
            f"pero el modelo produce {EMBEDDING_DIM}."
        )
    if config_indice.metric != "cosine":
        raise RuntimeError(
            f"El índice '{PINECONE_INDEX_NAME}' usa métrica '{config_indice.metric}', se esperaba 'cosine'."
        )

    print(
        f"[OK] Pinecone index conectado: {PINECONE_INDEX_NAME} "
        f"(dimensión {config_indice.dimension}, métrica {config_indice.metric})"
    )
    return indice


def ingest(chunks: list, api_key: str, limpiar_namespace: bool = False) -> PineconeVectorStore:
    """Ingiere los chunks en Pinecone y devuelve el PineconeVectorStore conectado.

    Los IDs son deterministas ({source}::{chunk_id}), así re-correr la ingesta
    actualiza los mismos vectores en lugar de duplicarlos.

    limpiar_antes=True borra el namespace completo antes de subir (útil cuando
    el corpus cambió y los chunk_ids ya no coinciden con los vectores viejos).
    """
    embeddings = get_embeddings()
    indice = get_pinecone_index(api_key)

    if limpiar_namespace:
        print(f"[..] Borrando namespace '{NAMESPACE}' ...")
        indice.delete(delete_all=True, namespace=NAMESPACE)
        print("[OK] Namespace vacío")

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=NAMESPACE,
        ids=[f"{c.metadata['source']}::{c.metadata['chunk_id']}" for c in chunks],
    )

    stats = indice.describe_index_stats()
    vectores_namespace = stats["namespaces"].get(NAMESPACE, {}).get("vector_count", 0)
    print("[OK] Vectores ingested")
    print(f"     Índice: {PINECONE_INDEX_NAME} | Dimensión: {EMBEDDING_DIM}")
    print(f"     Chunks generados: {len(chunks)}")
    print(f"     Vectores en el namespace: {vectores_namespace}")

    return vectorstore
