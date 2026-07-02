from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieve import HybridRetriever
from src.interface_utils import (
    build_filters,
    filter_options_from_rows,
    format_duration_ns,
    generation_mode_label,
    source_display_rows,
)
from src.ollama_adapter import (
    OllamaAdapterError,
    OllamaChatAdapter,
)
from src.ollama_rag import OllamaGroundedRAG


APP_TITLE = "Finance Knowledge RAG"

EXAMPLE_QUESTIONS = (
    "For Information Technology in August 2025, how far was actual expenditure below budget and below the pre-close forecast?",
    "A capital expenditure project is expected to cost AUD 300,000. Who must approve it, and what supporting documentation is required?",
    "How is net working capital calculated, what limit applies, and why does the KPI not have a universally higher-or-lower preferred direction?",
    "Which four Q4 priorities were agreed, and who owns the priority to recover gross margin?",
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_retriever() -> HybridRetriever:
    """Load the local embeddings and hybrid index once per app process."""
    return HybridRetriever(project_root=PROJECT_ROOT)


def initialise_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sources(result: dict[str, Any]) -> None:
    cited_sources = source_display_rows(result)

    if result.get("abstained"):
        st.caption(
            "No source is cited because the evidence gate abstained."
        )
        return

    if not cited_sources:
        st.warning("The answer contains no public source metadata.")
        return

    with st.expander(
        f"Cited sources ({len(cited_sources)})",
        expanded=False,
    ):
        for source in cited_sources:
            st.markdown(
                f"**[{source['source_number']}] "
                f"{source['document_title']}**"
            )
            st.caption(
                f"{source['file_type'].upper()} · "
                f"{source['citation']}"
            )
            st.code(
                f"Chunk: {source['chunk_id']}\n"
                f"Path: {source['source_path']}",
                language=None,
            )


def render_technical_details(
    result: dict[str, Any],
    wall_seconds: float | None,
) -> None:
    mode = str(result.get("generation_mode", "unknown"))
    metrics = result.get("provider_metrics") or {}

    columns = st.columns(4)
    columns[0].metric(
        "Mode",
        generation_mode_label(mode),
    )
    columns[1].metric(
        "Model",
        str(result.get("model") or "Not called"),
    )
    columns[2].metric(
        "Cited sources",
        int(result.get("cited_source_count", 0)),
    )
    columns[3].metric(
        "Wall time",
        f"{wall_seconds:.1f}s"
        if wall_seconds is not None
        else "Saved result",
    )

    with st.expander("Technical details", expanded=False):
        st.json(
            {
                "generation_mode": mode,
                "abstained": bool(result.get("abstained")),
                "citation_validation": result.get(
                    "citation_validation"
                ),
                "retrieved_source_count": result.get(
                    "source_count"
                ),
                "model_source_count": result.get(
                    "model_source_count"
                ),
                "provider_prompt_tokens": metrics.get(
                    "prompt_eval_count"
                ),
                "provider_output_tokens": metrics.get(
                    "eval_count"
                ),
                "provider_duration": format_duration_ns(
                    metrics.get("total_duration")
                ),
                "provider_error": result.get("provider_error"),
                "fallback_reason": result.get("fallback_reason"),
            }
        )

        model_context = str(result.get("model_context") or "")

        if model_context:
            st.markdown("**Evidence context sent to the model**")
            st.code(model_context, language=None)


def render_assistant_message(
    result: dict[str, Any],
    wall_seconds: float | None = None,
) -> None:
    answer = str(result.get("answer", "")).strip()

    if result.get("abstained"):
        st.warning(answer)
    else:
        st.markdown(answer)

    mode = str(result.get("generation_mode", "unknown"))

    if "fallback" in mode:
        st.info(
            "The guarded fallback path was used after validating "
            "the available evidence."
        )

    render_sources(result)
    render_technical_details(result, wall_seconds)


def ask_question(
    *,
    question: str,
    retriever: HybridRetriever,
    top_k: int,
    filters: dict[str, str],
) -> tuple[dict[str, Any], float]:
    adapter = OllamaChatAdapter()
    adapter.ensure_model_available()

    rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=adapter,
        top_k=top_k,
    )

    start = time.perf_counter()
    result = rag.answer(
        query=question,
        filters=filters,
    )
    wall_seconds = time.perf_counter() - start

    return result, wall_seconds


initialise_state()

st.title("📊 Finance Knowledge RAG")
st.caption(
    "A fully local finance and FP&A assistant using hybrid retrieval, "
    "grounded generation, verified citations and safe abstention."
)

try:
    with st.spinner("Loading the local hybrid retriever..."):
        retriever = load_retriever()
except Exception as exc:
    st.error(
        "The retriever could not be loaded. Confirm that the corpus, "
        "embeddings and model files exist in the project."
    )
    st.exception(exc)
    st.stop()

file_types, granularities, document_ids = filter_options_from_rows(
    retriever.rows
)

example_prompt: str | None = None

with st.sidebar:
    st.header("Local system")
    st.caption(
        "Generation v5.3 · Ollama · qwen3.5:4b · CPU"
    )

    if st.button(
        "Check Ollama and model",
        use_container_width=True,
    ):
        try:
            adapter = OllamaChatAdapter()
            adapter.ensure_model_available()
        except OllamaAdapterError as exc:
            st.error(str(exc))
        else:
            st.success(
                f"Ollama is available and "
                f"{adapter.model} is installed."
            )

    st.divider()
    st.subheader("Retrieval settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "The formal end-to-end evaluation used top_k=5 "
            "without metadata filters."
        ),
    )

    selected_file_type = st.selectbox(
        "File type",
        options=["All", *file_types],
    )

    selected_granularity = st.selectbox(
        "Granularity",
        options=["All", *granularities],
    )

    selected_document_id = st.selectbox(
        "Document",
        options=["All", *document_ids],
        help="Optional exact document filter.",
    )

    filters = build_filters(
        file_type=selected_file_type,
        granularity=selected_granularity,
        document_id=selected_document_id,
    )

    if filters or top_k != 5:
        st.warning(
            "These settings differ from the formally evaluated "
            "default: top_k=5 and no metadata filters."
        )
    else:
        st.success("Using the formally evaluated retrieval settings.")

    st.divider()
    st.subheader("Example questions")

    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(
            question,
            key=f"example_{index}",
            use_container_width=True,
        ):
            example_prompt = question

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    role = str(message["role"])

    with st.chat_message(role):
        if role == "user":
            st.markdown(str(message["content"]))
        else:
            render_assistant_message(
                result=message["result"],
                wall_seconds=message.get("wall_seconds"),
            )

typed_prompt = st.chat_input(
    "Ask a question about the finance knowledge base"
)
prompt = typed_prompt or example_prompt

if prompt:
    clean_prompt = prompt.strip()

    if not clean_prompt:
        st.warning("Enter a non-empty finance question.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(clean_prompt)

    with st.chat_message("assistant"):
        status = st.status(
            "Running the local grounded RAG...",
            expanded=True,
        )
        status.write("Retrieving hybrid dense and BM25 evidence.")

        try:
            result, wall_seconds = ask_question(
                question=clean_prompt,
                retriever=retriever,
                top_k=top_k,
                filters=filters,
            )

        except OllamaAdapterError as exc:
            status.update(
                label="Ollama is unavailable",
                state="error",
                expanded=True,
            )
            st.error(str(exc))
            st.info(
                "Start Ollama and confirm that qwen3.5:4b is installed."
            )

        except ValueError as exc:
            status.update(
                label="The request could not be processed",
                state="error",
                expanded=True,
            )
            st.error(str(exc))

        except Exception as exc:
            status.update(
                label="Unexpected local application error",
                state="error",
                expanded=True,
            )
            st.exception(exc)

        else:
            status.write(
                "Validating evidence, factual consistency and citations."
            )
            status.update(
                label="Grounded answer completed",
                state="complete",
                expanded=False,
            )

            render_assistant_message(
                result=result,
                wall_seconds=wall_seconds,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "result": result,
                    "wall_seconds": wall_seconds,
                }
            )
