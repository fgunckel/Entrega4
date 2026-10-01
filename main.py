# -*- coding: utf-8 -*-
"""Entry point de la Pre-Entrega 4 — Sistema de Retrieval Híbrido.

Uso:
    python main.py                     # demo: consulta de prueba + modo interactivo
    python main.py --ingest            # solo ingesta de los DOCX en Pinecone
    python main.py --evaluate          # evaluación (Precision@5 / Recall@5)
    python main.py --query "pregunta"  # consulta única
    python main.py --interactive       # solo modo interactivo

Sin argumentos: consulta de prueba + bucle interactivo.
NO hay capa de generación con LLM: el objetivo es inspeccionar el retrieval.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from src.config import TOP_K

# Consolas Windows (cp1252) no soportan emojis: forzamos UTF-8 en la salida
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def inicializar_sistema(limpiar_namespace: bool = False):
    """Inicializa el sistema completo: DOCX → chunks → embeddings → Pinecone → RAGSystem.

    Devuelve (rag_system, chunks). La ingesta usa IDs deterministas, así
    re-correrla actualiza los vectores en lugar de duplicarlos.
    Con limpiar_namespace=True borra el namespace antes de subir (re-ingesta
    completa, para cuando el corpus cambió).
    """
    from src.document_loader import load_and_split
    from src.ingestion_pipeline import ingest
    from src.rag_system import build_rag_system

    chunks = load_and_split()
    vectorstore = ingest(
        chunks,
        api_key=os.environ["PINECONE_API_KEY"],
        limpiar_namespace=limpiar_namespace,
    )
    rag_system = build_rag_system(chunks, vectorstore, k=TOP_K)
    return rag_system, chunks


def mostrar_resultados(resultados: list[dict]) -> None:
    """Muestra rank, fuente, clase y texto de cada fragmento recuperado."""
    print(f"\n📎 Top {len(resultados)} fragmentos recuperados:")
    for r in resultados:
        print(f"\n  {r['rank']}. [{r['fuente']} | clase {r['class']}]")
        print(f"     {r['contenido'][:200]}{'...' if len(r['contenido']) > 200 else ''}")
    print("-" * 80)


def mostrar_resultados_debug(resultados_debug: dict) -> None:
    """Imprime los resultados de diagnóstico separados por retriever.

    Muestra BM25, Vector y Hybrid para la misma consulta, con metadatos
    de cada fragmento (fuente/clase).
    """
    titulos = {
        "bm25": "BM25 RESULTS",
        "vector": "VECTOR RESULTS",
        "hybrid": "HYBRID RESULTS",
    }
    for nombre, titulo in titulos.items():
        print()
        print("=" * 60)
        print(titulo)
        print("=" * 60)
        for r in resultados_debug[nombre]:
            print(f"\n{r['rank']}. [{r['fuente']} | clase {r['class']}]")
            print(f"   {r['contenido'][:200]}{'...' if len(r['contenido']) > 200 else ''}")


def modo_interactivo(rag_system) -> None:
    """Bucle de preguntas: inspeccioná el retrieval con tus propias preguntas."""
    print("\n💬 Modo interactivo — consultá el retriever híbrido (BM25 + Pinecone)")
    print("   (escribí 'salir' para terminar)\n")

    while True:
        try:
            pregunta_usuario = input("🧑 Vos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Listo, terminamos la sesión.")
            break

        if pregunta_usuario.lower() in ("salir", "exit", "quit", ""):
            print("\n👋 Listo, terminamos la sesión.")
            break

        resultados = rag_system.retrieve(pregunta_usuario)

        if not resultados:
            print("⚠️ No se recuperó ningún fragmento.\n")
            continue

        mostrar_resultados(resultados)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sistema de Retrieval Híbrido — Pre-Entrega 4")
    parser.add_argument("--ingest", action="store_true", help="solo ingesta en Pinecone")
    parser.add_argument("--clean", action="store_true", help="con --ingest: borra el namespace antes de subir (re-ingesta completa)")
    parser.add_argument("--evaluate", action="store_true", help="corre la evaluación (Precision@5 / Recall@5)")
    parser.add_argument("--query", type=str, help="consulta única")
    parser.add_argument("--query-debug", type=str, help="consulta de diagnóstico: compara BM25 vs. Vector vs. Hybrid")
    parser.add_argument("--interactive", action="store_true", help="solo modo interactivo")
    args = parser.parse_args()

    load_dotenv()

    if not os.environ.get("PINECONE_API_KEY", "").strip():
        print(
            "[ERROR] Falta PINECONE_API_KEY. Definila en el archivo .env "
            "(gratis en https://app.pinecone.io)."
        )
        return 1

    print("[..] Inicializando sistema (carga + ingesta + retrievers)...")
    try:
        rag_system, chunks = inicializar_sistema(limpiar_namespace=args.clean)
    except Exception as exc:  # noqa: BLE001 - mostramos el error y abortamos, no silenciamos
        print(f"\n[ERROR] Falló la inicialización del sistema: {exc}")
        print("        Revisá tu PINECONE_API_KEY y tu conexión a internet.")
        return 1

    if args.ingest:
        print("\n[OK] Ingesta completada. Saliendo.")
        return 0

    if args.evaluate:
        from evaluation.evaluate import run_evaluation

        run_evaluation(rag_system)
        return 0

    if args.query:
        mostrar_resultados(rag_system.retrieve(args.query))
        return 0

    if args.query_debug:
        print(f"\nQuery:\n{args.query_debug}")
        mostrar_resultados_debug(rag_system.debug_query(args.query_debug))
        return 0

    # Demostración: consulta de prueba + modo interactivo
    pregunta_demo = "¿Qué es un namespace en Pinecone?"
    print(f"\n🔍 Consulta de prueba: {pregunta_demo}")
    mostrar_resultados(rag_system.retrieve(pregunta_demo))

    modo_interactivo(rag_system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
