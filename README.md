# Pre-Entrega 4 — AI Engineering Course Hybrid Retrieval System

## Project

Sistema de **retrieval híbrido** escalable para consultar el material del curso **AI Engineering** (Clases 1 a 4), escrito principalmente en español. Proyecto Python estructurado (no notebook).

## Objective

Implementar la Pre-Entrega 4: un sistema de recuperación escalable con **Pinecone Serverless** y **retrieval híbrido** (léxico + semántico), evaluado con un Golden Set.

> Este proyecto es principalmente de **RETRIEVAL**: no incluye capa de generación con LLM, agentes ni interfaz web. El objetivo es inspeccionar qué fragmentos recupera el sistema para cada pregunta.

## Arquitectura

```
Documentos (data/clase_1..4.docx)
      ↓  src/document_loader.py (Docx2txtLoader)
      ↓  RecursiveCharacterTextSplitter (chunk_size=600, chunk_overlap=100)
      ↓  Chunks (metadatos: source, class, chunk_id)
      │
      ├──────────────► BM25Retriever (k=5, sobre los mismos chunks)
      │
      ↓  src/ingestion_pipeline.py
      ↓  HuggingFace Embeddings (all-MiniLM-L6-v2, 384 dims)
      ↓  Pinecone Serverless (aws / us-east-1, métrica cosine)
      ↓  Vector Retriever (k=5)
      │
      └──────────────┐
                     ↓
              EnsembleRetriever (weights [0.5, 0.5])
                     ↓  src/rag_system.py (RAGSystem)
                  Top-5
                     ↓
        evaluation/evaluate.py → Precision@5 / Recall@5
```

## Estructura del proyecto

```
pre-entrega-4/
│
├── data/                    # 4 Knowledge Base del curso (DOCX)
├── src/
│   ├── config.py            # configuración centralizada
│   ├── document_loader.py   # carga DOCX + chunking + metadatos
│   ├── ingestion_pipeline.py# embeddings + Pinecone + ingesta
│   └── rag_system.py        # BM25 + vector + EnsembleRetriever
├── evaluation/
│   ├── golden_set.json      # Golden Set externalizado
│   └── evaluate.py          # Precision@5 / Recall@5
├── main.py                  # entry point ejecutable
├── requirements.txt
├── .env.example
├── .gitignore               # .env excluido de Git
└── README.md
```

## Dataset

Cuatro documentos enriquecidos del curso AI Engineering (Knowledge Base de las Clases 1 a 4), en español con terminología técnica en inglés preservada (ABC, Factory Pattern, streaming, retry, fallback, rate limit, LCEL, Runnable, embedding, chunking, vector database, BM25, namespace, metadata, Pinecone, Reciprocal Rank Fusion, Precision, Recall).

## Tecnologías

- Python
- LangChain (LCEL + retrievers)
- Pinecone Serverless
- HuggingFace Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensiones)
- BM25 (`rank-bm25` vía `BM25Retriever`)

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración de Pinecone

Los secretos **no** se hardcodean: se leen de variables de entorno.

1. Copiá `.env.example` a `.env`:
   ```bash
   cp .env.example .env
   ```
2. Editá `.env` con tu API key (gratis en https://app.pinecone.io):
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   ```

El archivo `.env` real está incluido en `.gitignore` y **nunca** se commitea.

## Ingesta

La ingesta (chunks → embeddings → Pinecone) corre automáticamente al iniciar el sistema, o de forma explícita:

```bash
python main.py --ingest
```

- Crea el índice Serverless `ai-engineering-course` (384 dims, cosine, aws/us-east-1) **solo si no existe**; si ya existe lo reutiliza.
- Los IDs de los vectores son deterministas (`clase_3.docx::17`), así que re-correr la ingesta **actualiza** los mismos vectores en lugar de duplicarlos.
- El texto se guarda en los metadatos del propio vector, así una sola consulta trae vector + texto + fuente.

## Retrieval

`src/rag_system.py` crea los tres componentes:

- **BM25**: retriever léxico sobre los mismos chunks usados para Pinecone (no se duplica el corpus). Fuerte con términos exactos, siglas y nombres propios ("2FA", "BM25", "RRF"), justo donde la búsqueda semántica es más débil.
- **Vector search**: retriever semántico sobre Pinecone (k=5, similitud coseno configurada en el índice). Fuerte con parafraseos.
- **EnsembleRetriever**: combina ambos con pesos baseline **BM25=0.5 / Vector=0.5**. No afirmamos que 0.5/0.5 sea óptimo — es la configuración de partida.

La clase `RAGSystem` encapsula el retriever híbrido y expone `retrieve(question, k=5)`. El nombre viene de la terminología del curso, pero esta Pre-Entrega se centra en retrieval: **no hay capa de generación con LLM**.

## Golden Set

`evaluation/golden_set.json` contiene 9 preguntas con documento fuente esperado (3 por clase), verificadas contra el corpus — cada respuesta está efectivamente en su documento esperado. El Golden Set está externalizado a JSON, fuera del código.

## Precision@5 / Recall@5

`evaluation/evaluate.py` implementa la evaluación:

- **Definición de relevancia**: un fragmento recuperado es *relevante* cuando su metadato `source` coincide con el documento esperado para esa pregunta. Es una evaluación simplificada a nivel documento, apropiada para el trabajo práctico — no una metodología universal de evaluación RAG.
- **Precision@5**: fracción de los 5 fragmentos recuperados cuyo `source` coincide con el esperado.
- **Recall@5**: 1 si el documento esperado aparece entre los recuperados, 0 si no. Como cada pregunta tiene un único documento esperado, solo puede valer 0 o 1.
- Las métricas se calculan de los resultados reales de retrieval — nada hardcodeado.

## Cómo ejecutar

```bash
# 1. Configurar la API key (una sola vez)
cp .env.example .env
#    y editá .env con tu PINECONE_API_KEY

# 2. Demo: consulta de prueba + modo interactivo
python main.py

# 3. Solo ingesta en Pinecone
python main.py --ingest

# 4. Evaluación (Precision@5 / Recall@5)
python main.py --evaluate

# 5. Consulta única
python main.py --query "¿Qué es LCEL?"

# 6. Solo modo interactivo
python main.py --interactive
```

`python main.py` sin argumentos inicializa el sistema (carga + ingesta + retrievers), ejecuta una consulta de prueba y abre el modo interactivo, que muestra para cada pregunta: **rank, fuente, clase y texto** de los top-5 fragmentos. Escribí `salir` para terminar.

## Notas de compatibilidad

- `EnsembleRetriever` se importa desde `langchain_classic.retrievers` (en versiones actuales de LangChain ya no está en `langchain.retrievers`).
- `BM25Retriever` se importa desde `langchain_community.retrievers`.
- `Docx2txtLoader` (de `langchain_community.document_loaders`) extrae texto de DOCX; requiere `docx2txt` (incluido en requirements).

## Seguridad

Las credenciales **no** se almacenan en el repositorio: la API key de Pinecone vive únicamente en la variable de entorno `PINECONE_API_KEY` (archivo `.env`, excluido de Git vía `.gitignore`). El repositorio incluye solo `.env.example` como plantilla.
