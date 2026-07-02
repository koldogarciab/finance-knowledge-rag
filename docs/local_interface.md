# Local Streamlit interface

The interface is a thin presentation layer over the evaluated Generation v5.3
pipeline. It does not duplicate retrieval, evidence selection, generation,
citation validation or abstention logic.

## Features

- Local chat interface using `OllamaGroundedRAG`
- Cached hybrid retriever and embedding model
- File type, granularity and document filters
- Formally evaluated default of `top_k=5`
- Grounded answers with numbered citations
- Safe abstention display
- Cited source metadata and evidence context
- Generation mode, model, latency and provider metrics
- Persistent conversation history for the current browser session
- Clear error messages when Ollama or the configured model is unavailable

## Install

Activate the existing environment from Anaconda PowerShell:

```powershell
& "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"
conda activate finance-rag
```

Install the updated requirements:

```powershell
python -m pip install -r requirements.txt
```

## Validate

```powershell
python -m py_compile app.py src\interface_utils.py
python tests\validate_streamlit_interface.py
python tests\validate_ollama_rag.py
```

## Run

Confirm that Ollama is running and that the evaluated model is installed:

```powershell
ollama list
```

Launch the interface from the repository root:

```powershell
streamlit run app.py
```

Streamlit will open the local application in the default browser. The terminal
must remain open while the application is running.

## Evaluation fidelity

The formal end-to-end evaluation used:

- Model: `qwen3.5:4b`
- Hybrid weights: dense `0.20`, lexical `0.80`
- RRF constant: `10`
- Retrieved chunks: `top_k=5`
- No metadata filters

The interface uses those defaults. Changing `top_k` or activating filters is
useful for exploration, but the resulting configuration is outside the frozen
formal evaluation.
