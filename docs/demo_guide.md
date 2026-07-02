# Demo guide

## Objective

Demonstrate three behaviours:

1. a normal grounded answer from Ollama;
2. a guarded fallback when generated output fails validation;
3. a pre-generation abstention for an unsupported question.

## Preparation

From Anaconda PowerShell:

```powershell
conda activate finance-rag
ollama list
streamlit run app.py
```

Keep the evaluated settings:

```text
Retrieved chunks: 5
File type: All
Granularity: All
Document: All
```

## Demo 1 — Normal grounded answer

Ask:

```text
A capital expenditure project is expected to cost AUD 300,000.
Who must approve it, and what supporting documentation is required?
```

Expected behaviour:

- mode: `Ollama`;
- answer includes Finance Director, CEO and Board of Directors;
- documentation includes benefits, cost, timing, risks and financial return;
- one or more numbered citations;
- source metadata can be expanded.

What to explain:

> The model does not answer from memory. It only receives the selected policy evidence. The citation validator confirms that the citation refers to the supplied context.

## Demo 2 — Guarded fallback

Ask:

```text
For Information Technology in August 2025, how far was actual expenditure
below budget and below the pre-close forecast?
```

Expected evidence:

- budget: AUD 390,000;
- actual: AUD 382,200;
- favourable variance: AUD 7,800 / 2.00%;
- forecast: AUD 383,760;
- actual below forecast: AUD 1,560.

The mode may be `Extractive fallback`.

What to explain:

> Ollama was called, but its output did not pass every validation. Because the evidence gate had already confirmed sufficient support, the system returned a citation-preserving extractive answer instead of an unsafe model answer.

## Demo 3 — Unsupported question

Ask:

```text
What effective corporate tax rate was forecast for Harbour Retail Group for FY2025/26?
```

Expected behaviour:

- mode: `Retrieval abstention`;
- no sources cited;
- response in a fraction of a second;
- Ollama not called.

What to explain:

> The corpus contains general tax references but not the requested forecast tax rate. The grounding gate rejects the question before generation, preventing a plausible but invented answer.

## Optional interface features

Show:

- source metadata;
- model context;
- generation mode;
- provider latency;
- document and format filters.

Clarify that filters and non-default `top_k` values are exploratory and were not part of the frozen end-to-end benchmark.

## Demo checklist

- Ollama running
- `qwen3.5:4b` installed
- correct conda environment
- default retrieval settings
- browser zoom suitable for the screen
- terminal kept open
- no confidential data shown
