# DriveWise — Metadata-Aware Automotive RAG Assistant

DriveWise is a brochure-grounded conversational assistant that answers natural-language
questions about specific car brands and models using Retrieval-Augmented Generation (RAG).
Every answer is generated strictly from retrieved brochure content, with sources attributed
back to the originating document, section, and page — no answers come from the language
model's general knowledge.

## Pipeline 
Brand Selection → Model Selection → User Question → Metadata Filtering →
FAISS Vector Retrieval → Cross-Encoder Re-ranking → Prompt Construction →
LLM Generation → Grounded Answer + Source Attribution → Logging

## Features

- **Metadata-filtered retrieval** — similarity search runs only over the selected
  brand/model's chunks, not the whole corpus.
- **Structured, section-aware chunking** — brochure content is classified into sections
  (engine, safety, dimensions, comfort, infotainment, etc.) via keyword-frequency
  classification.
- **Cross-encoder re-ranking** — retrieved chunks are re-scored against the query for
  better precision than embedding similarity alone.
- **Source attribution** — every answer cites the document, section, and page it came from.
- **Structured logging** — every query is logged with per-stage latency (retrieval,
  re-ranking, generation) in JSONL format.
- **Streamlit demo** — interactive brand/model selection and Q&A interface.

## Tech Stack

- Python 3.12
- `PyMuPDF` — PDF parsing
- `sentence-transformers` (`BAAI/bge-small-en-v1.5`) — embeddings
- `FAISS` (`IndexFlatIP`) — vector store
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — re-ranking
- Google Gemini API (`gemini-3.1-flash-lite`) — answer generation
- `Streamlit` — demo UI

## Project Structure
   
DriveWise/
├── data/brochures/<BRAND>/<Model>.pdf   # source brochures
├── notebooks/DriveWise.ipynb            # full pipeline, built section by section
├── app/app.py                           # Streamlit demo
├── vectorstore/                         # persisted FAISS index + chunk metadata
├── logs/query_log.jsonl                 # structured query logs
├── evaluation/                          # evaluation outputs
├── requirements.txt
└── .env                                 # GOOGLE_API_KEY (not committed)

## Setup

1. Clone the repo and install dependencies:
```bash
   pip install -r requirements.txt
```
2. Create a `.env` file in the project root:

GOOGLE_API_KEY=your_api_key_here

3. Place brochure PDFs under `data/brochures/<BRAND>/<Model>.pdf`.
4. Run `notebooks/DriveWise.ipynb` top to bottom to build the vector store.
5. Launch the demo:
```bash
   streamlit run app/app.py
```

## Evaluation

The notebook includes a lightweight, framework-free evaluation (Section 13) that runs a
set of representative questions through the full pipeline and reports per-stage latency
and success rate.

## Limitations

- Section classification uses keyword frequency, not a learned classifier — it can
  misclassify content outside the predefined keyword lists.
- Chunking is currently page-level; long or multi-topic pages aren't split further.
- Brochures vary significantly in length/content density (e.g. a 3-page brochure yields
  far fewer retrievable chunks than an 11-page one) — retrieval quality scales with
  source document richness.
- Evaluation measures latency and success rate, not answer correctness or faithfulness
  (no automated framework like Ragas is used).

## Future Improvements

- Finer-grained, section-aware chunking for long/dense pages.
- Automated correctness/faithfulness evaluation with curated ground-truth answers.
- Caching embeddings per brochure to avoid full re-processing on every run.


























