# -*- coding: utf-8 -*-
"""Evaluación de la Pre-Entrega 4: Precision@5 y Recall@5 sobre el Golden Set.

Refactorizado de la lógica del notebook (misma metodología, mismo k=5).

Definición de relevancia (evaluación simplificada a nivel documento):
un fragmento recuperado es relevante cuando su metadato 'source' coincide
con el documento esperado para esa pregunta. Con un único documento
relevante posible, Recall@5 solo puede valer 0 o 1.

Es una evaluación documento-level apropiada para el trabajo práctico,
no una metodología universal de evaluación RAG.
"""

import json

from src.config import GOLDEN_SET_PATH, TOP_K


def load_golden_set(path=None) -> list[dict]:
    """Carga el Golden Set desde evaluation/golden_set.json."""
    path = path or GOLDEN_SET_PATH
    with open(path, encoding="utf-8") as f:
        golden_set = json.load(f)

    if not golden_set:
        raise RuntimeError(f"El Golden Set está vacío: {path}")

    print(f"[OK] Golden Set cargado: {len(golden_set)} preguntas ({path.name})")
    return golden_set


def evaluar(rag_system, golden_set: list[dict]) -> dict:
    """Calcula Precision@5 y Recall@5 por pregunta y sus promedios.

    Para cada pregunta:
    1. Ejecuta retrieval (top-5).
    2. Lee el metadato 'source' de cada fragmento recuperado.
    3. Un fragmento es relevante si su 'source' coincide con el documento esperado.
    4. Precision@5 = fracción de los 5 recuperados cuyo source coincide con el esperado.
    5. Recall@5 = 1 si el documento esperado aparece entre los recuperados, 0 si no.

    Las métricas se calculan de los resultados reales — nada hardcodeado.
    """
    resultados_por_pregunta = []

    for caso in golden_set:
        top_k = rag_system.retrieve(caso["pregunta"])
        fuentes_recuperadas = [r["fuente"] for r in top_k]

        # Recall@5: ¿el documento esperado aparece entre los recuperados? (0 o 1)
        recall = 1.0 if caso["documento_id_esperado"] in fuentes_recuperadas else 0.0

        # Precision@5: qué fracción de los 5 recuperados coinciden con la fuente esperada
        coincidencias = sum(1 for f in fuentes_recuperadas if f == caso["documento_id_esperado"])
        precision = coincidencias / len(top_k) if top_k else 0.0

        resultados_por_pregunta.append({
            "pregunta": caso["pregunta"],
            "esperado": caso["documento_id_esperado"],
            "recuperados": fuentes_recuperadas,
            "recall@5": recall,
            "precision@5": precision,
        })

    recall_promedio = sum(r["recall@5"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)
    precision_promedio = sum(r["precision@5"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)

    return {
        "detalle": resultados_por_pregunta,
        "recall@5_promedio": recall_promedio,
        "precision@5_promedio": precision_promedio,
    }


def imprimir_reporte(reporte: dict) -> None:
    """Imprime resultados individuales y agregados."""
    print("=" * 80)
    for r in reporte["detalle"]:
        estado = "✅" if r["recall@5"] == 1.0 else "❌"
        print(f"{estado} {r['pregunta']}")
        print(f"   Esperado: {r['esperado']} | Recuperados: {r['recuperados']}")
        print(f"   Recall@5: {r['recall@5']:.0%} | Precision@5: {r['precision@5']:.0%}\n")

    print("=" * 80)
    print(f"📊 RECALL@5 PROMEDIO:    {reporte['recall@5_promedio']:.1%}")
    print(f"📊 PRECISION@5 PROMEDIO: {reporte['precision@5_promedio']:.1%}")


def run_evaluation(rag_system, golden_set_path=None) -> dict:
    """Punto de entrada: carga el Golden Set, evalúa y muestra el reporte."""
    golden_set = load_golden_set()
    reporte = evaluar(rag_system, golden_set)
    imprimir_reporte(reporte)
    return reporte


if __name__ == "__main__":
    # Permite ejecutar la evaluación directamente: python -m evaluation.evaluate
    # (requiere que el sistema ya esté inicializado — usar main.py --evaluate)
    raise SystemExit(
        "Ejecutá la evaluación con: python main.py --evaluate "
        f"(top_k={TOP_K})"
    )
