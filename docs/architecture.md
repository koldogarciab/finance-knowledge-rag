# Technical architecture

## Design principles

The system was built around five principles:

1. **Local-first:** embeddings, retrieval, generation and interface run locally.
2. **Separation of concerns:** ingestion, retrieval, grounding, generation and presentation are independent layers.
3. **Evidence gating:** retrieval success alone is not sufficient; selected evidence must also satisfy the query.
4. **Traceability:** every supported answer exposes source metadata and validated citation numbers.
5. **Safe degradation:** invalid model output is repaired, replaced by a guarded fallback or rejected.

## Component view

```mermaid
flowchart TB
    subgraph Data
        A1[Corpus blueprint]
        A2[Generated raw files]
        A3[Normalised records]
        A4[Token-aware chunks]
        A5[Embedding index]
    end

    subgraph Retrieval
        B1[Dense MiniLM search]
        B2[BM25 search]
        B3[Weighted RRF]
        B4[Metadata filters]
    end

    subgraph Grounded generation
        C1[Question-specific source selection]
        C2[Evidence gate]
        C3[Prompt construction]
        C4[Ollama qwen3.5:4b]
        C5[Citation validation]
        C6[Financial consistency checks]
        C7[Repair / extractive fallback]
        C8[Pre-generation abstention]
    end

    subgraph Presentation
        D1[Streamlit]
        D2[Answer and citations]
        D3[Source metadata]
        D4[Technical diagnostics]
    end

    A1 --> A2 --> A3 --> A4 --> A5
    A4 --> B2
    A5 --> B1
    B1 --> B3
    B2 --> B3
    B4 --> B3
    B3 --> C1 --> C2
    C2 -->|supported| C3 --> C4 --> C5 --> C6
    C2 -->|unsupported| C8
    C6 -->|valid| D1
    C6 -->|invalid| C7 --> D1
    C8 --> D1
    D1 --> D2
    D1 --> D3
    D1 --> D4
```

## Data layer

The corpus is generated from a controlled blueprint and exported to five formats:

- management report in PDF;
- policies and procedures in DOCX;
- monthly budget-versus-actual data in CSV;
- KPI dictionary in JSON;
- forecast meeting notes in Markdown.

Extraction converts the files into a common record schema. Token-aware chunking then creates retrieval units while preserving file-specific metadata such as page, section, period, department and granularity.

## Embedding layer

`sentence-transformers` produces 384-dimensional MiniLM embeddings. The vectors are stored as `float32` and normalised so dense similarity can use the dot product as cosine similarity.

## Retrieval layer

The hybrid retriever creates:

- a dense ranking from the embedding index;
- a lexical BM25 ranking over `retrieval_text`;
- a fused ranking through weighted reciprocal rank fusion.

The selected lexical-heavy weighting reflects the corpus: finance questions often contain exact entity names, months, KPI labels, thresholds and policy terms that benefit from lexical matching.

Metadata filters are available, but the frozen end-to-end evaluation uses global retrieval without filters.

## Grounding layer

The grounding layer performs more than top-k retrieval:

1. identifies question intent and requested facts;
2. prioritises candidate sources;
3. selects the evidence units supplied to the model;
4. rejects unsupported requests before generation;
5. preserves contiguous citation numbering.

This layer prevents the local model from seeing irrelevant corpus content.

## Generation and validation

Ollama receives:

- the user question;
- explicit grounded-answer rules;
- only the selected numbered evidence.

The output is checked for:

- citation presence and validity;
- sentence-level citation coverage;
- numeric completeness;
- expense and forecast polarity;
- zero-variance wording;
- approval completeness;
- requested difference completeness;
- context-dependent KPI explanations.

If the output fails, the pipeline may retry with a repair prompt. If sufficient evidence exists but model output remains invalid, a guarded extractive fallback is used. If the evidence is insufficient, the system abstains.

## Interface layer

`app.py` calls the evaluated `OllamaGroundedRAG` class directly. It does not reimplement the RAG logic.

The interface exposes:

- answer;
- citation list;
- generation mode;
- model;
- wall time;
- provider metrics;
- selected model context.

## Security and production considerations

The current project is a local portfolio implementation. A production deployment would additionally need:

- document-level access control;
- user authentication and audit logs;
- encrypted storage;
- source freshness and document lifecycle controls;
- PII and confidential-data handling;
- monitoring for retrieval drift;
- benchmark expansion with real domain users.
