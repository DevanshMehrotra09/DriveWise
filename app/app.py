# ============================================================
# DriveWise Streamlit Application
# ============================================================

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

import google.generativeai as genai

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)
# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
LOGS_DIR = PROJECT_ROOT / "logs"

FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "chunk_metadata.parquet"

LOG_FILE = LOGS_DIR / "query_log.jsonl"
# ============================================================
# Environment & Gemini Configuration
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not found.\n"
        "Create a .env file containing:\n"
        "GOOGLE_API_KEY=your_api_key"
    )

genai.configure(api_key=GOOGLE_API_KEY)

# ============================================================
# Model Configuration
# ============================================================

LLM_MODEL_NAME = "gemini-3.5-flash"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

llm = genai.GenerativeModel(LLM_MODEL_NAME)
# ============================================================
# Cached Model Loading
# ============================================================

@st.cache_resource
def load_models():
    """
    Load the embedding model and reranker once per Streamlit session.
    """
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return embedder, reranker


# Load models
embedder, reranker = load_models()
# ============================================================
# Cached Vector Store Loading
# ============================================================

@st.cache_resource
def load_vector_store():
    """
    Load the FAISS index and metadata created by the notebook.
    """
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = pd.read_parquet(METADATA_PATH)

    return index, metadata


# Load vector store
faiss_index, chunk_metadata_df = load_vector_store()
# ============================================================
# Retrieval Functions
# ============================================================

def filter_chunk_indices(
    metadata: pd.DataFrame,
    brand: str,
    model: str,
) -> np.ndarray:
    """
    Return metadata row indices belonging to the selected brand/model.
    """
    mask = (
        (metadata["brand"] == brand) &
        (metadata["model"] == model)
    )

    return metadata[mask].index.to_numpy()


def embed_query(query: str) -> np.ndarray:
    """
    Embed a user query using the BGE embedding model.
    """
    prefixed_query = BGE_QUERY_INSTRUCTION + query

    query_embedding = embedder.encode(
        [prefixed_query],
        convert_to_numpy=True,
    )

    faiss.normalize_L2(query_embedding)

    return query_embedding.astype("float32")


def retrieve(
    query: str,
    brand: str,
    model: str,
    index: faiss.Index,
    metadata: pd.DataFrame,
    top_k: int = 10,
):
    """
    Retrieve the top brochure chunks after metadata filtering.
    """

    allowed_indices = set(
        filter_chunk_indices(
            metadata,
            brand,
            model,
        ).tolist()
    )

    if not allowed_indices:
        return (
            pd.DataFrame(columns=list(metadata.columns) + ["similarity_score"]),
            0.0,
        )

    query_embedding = embed_query(query)

    start_time = time.perf_counter()

    scores, indices = index.search(
        query_embedding,
        index.ntotal,
    )

    retrieval_time = time.perf_counter() - start_time

    scores = scores[0]
    indices = indices[0]

    filtered = [
        (idx, score)
        for idx, score in zip(indices, scores)
        if idx in allowed_indices
    ][:top_k]

    if not filtered:
        return (
            pd.DataFrame(columns=list(metadata.columns) + ["similarity_score"]),
            retrieval_time,
        )

    rows, similarity_scores = zip(*filtered)

    retrieved = metadata.loc[list(rows)].copy()

    retrieved["similarity_score"] = similarity_scores

    return retrieved.reset_index(drop=True), retrieval_time
# ============================================================
# Re-ranking
# ============================================================

def rerank(
    query: str,
    retrieved: pd.DataFrame,
    top_n: int = 4,
):
    """
    Re-rank retrieved chunks using the cross-encoder.
    """

    if retrieved.empty:
        return retrieved, 0.0

    pairs = [
        (query, row.text)
        for row in retrieved.itertuples()
    ]

    start_time = time.perf_counter()

    scores = reranker.predict(pairs)

    rerank_time = time.perf_counter() - start_time

    reranked = retrieved.copy()

    reranked["rerank_score"] = scores

    reranked = (
        reranked
        .sort_values(
            "rerank_score",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return reranked, rerank_time
# ============================================================
# Prompt Template
# ============================================================

PROMPT_TEMPLATE = """You are DriveWise, an assistant that answers questions about the {brand} {model} \
using ONLY the brochure excerpts provided below.

Rules:
- Answer strictly from the provided context. Do not use outside knowledge.
- If the context does not contain the answer, say so explicitly — do not guess or hallucinate.
- Keep the answer concise and directly relevant to the question.
- When useful, mention which section/page the information came from.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(
    question: str,
    brand: str,
    model: str,
    context_chunks: pd.DataFrame,
) -> str:
    """
    Build the grounded prompt sent to Gemini.
    """

    context = "\n\n".join(
        f"[Section: {row.section} | Page: {row.page}]\n{row.text}"
        for row in context_chunks.itertuples()
    )

    return PROMPT_TEMPLATE.format(
        brand=brand,
        model=model,
        context=context,
        question=question,
    )
# ============================================================
# RAG Response Container
# ============================================================

@dataclass
class RagResponse:
    answer: str
    sources: list[dict]
    retrieval_time: float
    rerank_time: float
    generation_time: float
    status: str


# ============================================================
# End-to-End RAG Pipeline
# ============================================================

def ask_drivewise(
    question: str,
    brand: str,
    model: str,
    index: faiss.Index,
    metadata: pd.DataFrame,
    top_k: int = 10,
    top_n: int = 4,
) -> RagResponse:

    retrieved, retrieval_time = retrieve(
        question,
        brand,
        model,
        index,
        metadata,
        top_k=top_k,
    )

    if retrieved.empty:
        return RagResponse(
            answer=f"No brochure data found for {brand} {model}.",
            sources=[],
            retrieval_time=retrieval_time,
            rerank_time=0.0,
            generation_time=0.0,
            status="no_context",
        )

    reranked, rerank_time = rerank(
        question,
        retrieved,
        top_n=top_n,
    )

    prompt = build_prompt(
        question,
        brand,
        model,
        reranked,
    )

    start_time = time.perf_counter()

    try:
        response = llm.generate_content(prompt)
        answer = response.text
        status = "success"

    except Exception as exc:
        error = str(exc)

        if "429" in error or "RESOURCE_EXHAUSTED" in error:
            answer = (
                "Gemini API quota exceeded. "
                "Please wait for quota refresh and try again."
            )
            status = "rate_limited"

        else:
            answer = f"Generation failed:\n\n{exc}"
            status = "error"

    generation_time = time.perf_counter() - start_time

    sources = [
        {
            "document": row.document_name,
            "page": int(row.page),
            "section": row.section,
        }
        for row in reranked.itertuples()
    ]

    return RagResponse(
        answer=answer,
        sources=sources,
        retrieval_time=retrieval_time,
        rerank_time=rerank_time,
        generation_time=generation_time,
        status=status,
    )
# ============================================================
# Query Logging
# ============================================================

def log_interaction(
    question: str,
    brand: str,
    model: str,
    response: RagResponse,
) -> None:
    """
    Log each query and its performance metrics.
    """

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_entry ={
    "timestamp": time.time(),
    "query": question,
    "brand": brand,
    "model": model,
    "response_time_s": round(
        response.retrieval_time
        + response.rerank_time
        + response.generation_time,
        4,
    ),
    "retrieval_latency_s": round(response.retrieval_time, 4),
    "rerank_latency_s": round(response.rerank_time, 4),
    "generation_latency_s": round(response.generation_time, 4),
    "retrieved_chunks": response.sources,
    "answer_status": response.status,
    "failure_status": response.status in ("error", "rate_limited"),
}

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(log_entry) + "\n")
        # ============================================================
# Streamlit User Interface
# ============================================================

st.set_page_config(
    page_title="DriveWise",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 DriveWise")
st.caption("Brochure-Grounded Conversational AI for Cars")

brand = st.selectbox(
    "Select Brand",
    sorted(chunk_metadata_df["brand"].unique()),
)

model = st.selectbox(
    "Select Model",
    sorted(
        chunk_metadata_df[
            chunk_metadata_df["brand"] == brand
        ]["model"].unique()
    ),
)

question = st.text_input(
    "Ask a question about this vehicle",
    placeholder="e.g. How many airbags does this car have?",
)

if st.button("Ask DriveWise"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching brochure..."):

        response = ask_drivewise(
            question,
            brand,
            model,
            faiss_index,
            chunk_metadata_df,
        )

        log_interaction(
            question,
            brand,
            model,
            response,
        )

    st.subheader("Answer")
    st.write(response.answer)

    st.subheader("Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Retrieval",
        f"{response.retrieval_time*1000:.1f} ms",
    )

    col2.metric(
        "Reranking",
        f"{response.rerank_time*1000:.1f} ms",
    )

    col3.metric(
        "Generation",
        f"{response.generation_time*1000:.1f} ms",
    )

    if response.sources:

        st.subheader("Sources")

        for src in response.sources:

            st.markdown(
                f"- **{src['document']}** | "
                f"Section: **{src['section']}** | "
                f"Page **{src['page']}**"
            )