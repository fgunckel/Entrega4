# -*- coding: utf-8 -*-
"""Sistema de retrieval híbrido (RAGSystem).

Crea: 1) BM25Retriever, 2) Pinecone vector retriever, 3) EnsembleRetriever
con pesos baseline BM25=0.5 / Vector=0.5.

El nombre RAGSystem viene de la terminología del curso, pero esta Pre-Entrega
se centra en retrieval: NO hay capa de generación con LLM.
"""

from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from src.config import NAMESPACE, TOP_K


def _docs_a_resultados(docs: list) -> List[dict]:
    """Convierte Documents en dicts con rank + metadatos (lógica compartida)."""
    return [
        {
            "rank": i,
            "contenido": d.page_content,
            "fuente": d.metadata.get("source", "desconocida"),
            "class": d.metadata.get("class", "desconocida"),
            "chunk_id": d.metadata.get("chunk_id", "desconocido"),
        }
        for i, d in enumerate(docs, 1)
    ]


class RAGSystem:
    """Encapsula el retriever híbrido y expone un método simple de retrieval.

    Para diagnóstico también puede retener los retrievers individuales
    (bm25_retriever, vector_retriever) sin alterar el path de producción,
    que siempre usa hybrid_retriever.
    """

    def __init__(self, retriever, k: int = TOP_K, bm25_retriever=None, vector_retriever=None):
        self.hybrid_retriever = retriever
        self.retriever = retriever  # alias de producción (no cambiar el path default)
        self.k = k
        # Opcionales: solo para el modo diagnóstico (--query-debug)
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever

    def retrieve(self, question: str) -> List[dict]:
        """Recupera el top-k de fragmentos para una pregunta (path de producción).

        Devuelve una lista de dicts con rank, contenido, fuente, clase y chunk_id.
        """
        docs = self.hybrid_retriever.invoke(question)[: self.k]
        return _docs_a_resultados(docs)

    def debug_query(self, question: str) -> dict:
        """Ejecuta la MISMA consulta independientemente contra los tres retrievers.

        Función de observabilidad: permite comparar BM25 vs. Vector vs. Hybrid
        para la misma query y determinar qué etapa produce los resultados
        inesperados. No altera el retrieval de producción.

        Devuelve {"bm25": [...], "vector": [...], "hybrid": [...]}.
        """
        resultados = {}
        for nombre, retriever in (
            ("bm25", self.bm25_retriever),
            ("vector", self.vector_retriever),
            ("hybrid", self.hybrid_retriever),
        ):
            if retriever is None:
                raise RuntimeError(
                    f"debug_query requiere el retriever '{nombre}'. "
                    "Usá build_rag_system() para inicializar el sistema."
                )
            docs = retriever.invoke(question)[: self.k]
            resultados[nombre] = _docs_a_resultados(docs)
        return resultados


def build_bm25_retriever(chunks: list) -> BM25Retriever:
    """Retriever léxico sobre los mismos chunks usados para Pinecone.

    No se duplica el corpus. BM25 es fuerte con términos exactos, siglas y
    nombres propios ("2FA", "BM25", "RRF"), justo donde la búsqueda semántica
    es más débil — por eso se combinan.
    """
    retriever_bm25 = BM25Retriever.from_documents(chunks)
    retriever_bm25.k = TOP_K
    print("[OK] BM25 Retriever ready")
    return retriever_bm25


def build_vector_retriever(vectorstore):
    """Retriever semántico sobre Pinecone (k=5, similitud coseno del índice)."""
    retriever_vectorial = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K, "namespace": NAMESPACE},
    )
    print("[OK] Vector Retriever ready")
    return retriever_vectorial


def build_hybrid_retriever(retriever_bm25, retriever_vectorial) -> EnsembleRetriever:
    """Combina BM25 + vectorial con EnsembleRetriever.

    Pesos baseline del curso: [0.5, 0.5]. No afirmamos que 0.5/0.5 sea óptimo
    — es la configuración de partida.
    """
    retriever_hibrido = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vectorial],
        weights=[0.5, 0.5],
    )
    print("[OK] Hybrid Retriever ready")
    return retriever_hibrido


def build_rag_system(chunks: list, vectorstore, k: int = TOP_K) -> RAGSystem:
    """Punto de entrada: crea los tres retrievers y el RAGSystem híbrido."""
    retriever_bm25 = build_bm25_retriever(chunks)
    retriever_vectorial = build_vector_retriever(vectorstore)
    retriever_hibrido = build_hybrid_retriever(retriever_bm25, retriever_vectorial)
    return RAGSystem(
        retriever_hibrido,
        k=k,
        bm25_retriever=retriever_bm25,
        vector_retriever=retriever_vectorial,
    )
