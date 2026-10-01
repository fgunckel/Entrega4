# -*- coding: utf-8 -*-
"""Configuración centralizada del proyecto Pre-Entrega 4.

Todas las constantes del proyecto viven acá. Centralizar EMBEDDING_DIM evita
el clásico error de crear un índice de 1536 dimensiones (específico de OpenAI)
cuando nuestro modelo local produce vectores de 384.
"""

from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVALUATION_DIR = BASE_DIR / "evaluation"
GOLDEN_SET_PATH = EVALUATION_DIR / "golden_set.json"

# Pinecone
PINECONE_INDEX_NAME = "ai-engineering-course"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
NAMESPACE = "curso-ai-engineering"

# Embeddings: un solo modelo para ingesta y consulta (homogeneidad de embeddings)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # el índice de Pinecone debe coincidir exactamente

# Chunking (configuración baseline del curso — no optimizada)
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5
