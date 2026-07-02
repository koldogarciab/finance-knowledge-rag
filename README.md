# finance-knowledge-rag

A multiformat, fully local RAG assistant for synthetic finance and FP&A
documentation.

The project combines dense MiniLM embeddings, BM25 lexical retrieval, weighted
reciprocal rank fusion, deterministic evidence selection, local generation
with Ollama, verified citations and safe abstention.

## Local interface

The Streamlit application is a thin layer over the evaluated Generation v5.3
pipeline.

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The evaluated local model is `qwen3.5:4b`. Ollama must be running before a
question is submitted.

See [`docs/local_interface.md`](docs/local_interface.md) for setup,
validation and usage details.

## Formal generation evaluation

The final frozen evaluation includes 30 supported finance questions and
10 unsupported questions.

- 30/30 supported questions answered
- 10/10 unsupported questions correctly rejected before model generation
- 100% valid citation rate
- 100% expected evidence retrieved, sent to the model and cited
- 0 automatic failures
- 29 answers accepted after automatic and manual review, with 1 conservative
  partial answer

Detailed reports are stored in `reports/`.
