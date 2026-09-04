# IPOlytics: Project and Interview Guide

## 1. One-minute explanation

**IPOlytics** is a portfolio project with two AI features:

1. It predicts whether an IPO is likely to have a positive or negative listing using a pre-trained machine-learning classification pipeline.
2. It answers questions about an already indexed IPO prospectus using retrieval-augmented generation (RAG). It retrieves relevant PDF passages from FAISS and asks Gemini to answer only from those passages, returning source page numbers.

The user interface is Streamlit. FastAPI provides the backend APIs. The ML model and RAG resources are loaded once and reused instead of being loaded in the Streamlit process.

## 2. Architecture

```text
                    Streamlit UI
                         |
          +--------------+--------------+
          |                             |
     POST /predict                    POST /chat
          |                             |
       FastAPI                      FastAPI
          |                             |
 PredictionService                RAGPipeline
          |                             |
 joblib ML pipeline          query embedding (MiniLM)
                                        |
                                  persisted FAISS index
                                        |
                                top relevant PDF chunks
                                        |
                                     Gemini API
                                        |
                            answer + filename + page number
```

## 3. Important request flows

### IPO prediction

1. The user enters the IPO date, issue size, subscription values, total subscription, and offer price.
2. Streamlit sends JSON to `POST /predict`.
3. Pydantic validates the request schema.
4. `PredictionService` uses `engineer_features()` to create the same 16 features used during training.
5. The serialized joblib model returns a class and class probabilities.
6. FastAPI returns the label and positive/negative probabilities; Streamlit displays them.

### IPO research question

1. The user enters a question in Streamlit.
2. Streamlit sends `{"question": "..."}` to `POST /chat`.
3. FastAPI reuses one cached `RAGPipeline` for the process.
4. The sentence-transformer creates an embedding only for the **question**.
5. FAISS searches the already persisted index and returns the top relevant chunks.
6. The backend sends compact, sentence-complete excerpts to Gemini with grounding instructions.
7. Gemini returns an answer, and FastAPI returns it with PDF filename/page sources.

**Key point to say in an interview:** the 616-page PDF is not read, chunked, embedded, or indexed during `POST /chat`. Only the short user question is embedded. FAISS retrieval was measured at about 0.10 seconds; Gemini is usually the slowest step because it is a remote API call.

## 4. File-by-file guide

| File / folder | What it does | What you should know |
|---|---|---|
| `frontend/streamlit_app.py` | Simple two-tab Streamlit interface. | It calls the FastAPI APIs with `requests`; it does not load ML, FAISS, or Gemini directly. |
| `app/main.py` | FastAPI application and route definitions. | Starts RAG resources once, exposes `/health`, `/predict`, and `/chat`, and maps errors to useful HTTP responses. |
| `app/schemas/prediction.py` | Pydantic request/response schemas for prediction. | Validation protects the API from invalid values before model inference. |
| `app/schemas/chat.py` | Pydantic schemas for a question, answer, and source metadata. | Each source contains a filename, page, and similarity score. |
| `app/services/prediction_service.py` | Loads and calls the saved classifier. | Uses a singleton so the joblib model is loaded once per backend process. |
| `app/ml/preprocessing.py` | Feature engineering for prediction. | Feature order must exactly match model training. It creates date, ratio, log, and difference features. |
| `app/rag/rag_pipeline.py` | Orchestrates a question-answering query. | Loads persisted FAISS only; embeds query -> retrieves chunks -> prompts Gemini -> returns citations. |
| `app/rag/embeddings.py` | SentenceTransformer wrapper. | `all-MiniLM-L6-v2` creates normalized query/document vectors. The model is loaded once and uses local cache at runtime. |
| `app/rag/vector_store.py` | FAISS index and metadata persistence/search. | `index.faiss` stores vectors; `metadata.pkl` stores each chunk's text, filename, and page. |
| `app/rag/retriever.py` | Retrieval layer. | Embeds one question and sends it to FAISS; current RAG pipeline retrieves two chunks. |
| `app/rag/prompts.py` | Gemini grounding prompt template. | Tells Gemini to use only retrieved context, be concise, and cite pages. |
| `app/rag/llm.py` | Gemini wrapper. | Converts Gemini quota and deadline errors into clear backend errors. |
| `app/rag/document_loader.py` | PDF text extraction utility using PyMuPDF (`fitz`). | Intended for offline ingestion/index creation, not called during chat. |
| `app/rag/chunker.py` | Splits extracted page text into overlapping chunks. | Used during offline indexing so retrieval is more precise than whole-page search. |
| `data/documents/Tempsens-Instruments-India-Limited-Prospectus.pdf` | Demo IPO prospectus. | The single source document for the portfolio demo. |
| `vectorstore/index.faiss` | Persisted FAISS vector index. | Loaded at backend startup; not rebuilt on questions. |
| `vectorstore/metadata.pkl` | Persisted chunk metadata. | Lets the app display source document names and page numbers. |
| `ml/best_classification_pipeline.pkl` | Trained prediction model artifact. | Loaded with joblib; do not retrain or load it in Streamlit. |
| `ml/feature_config.json` | Model/feature configuration artifact. | Keep it aligned with the training pipeline. |
| `tests/test_prediction.py` | Prediction-related tests. | Demonstrates validation and preprocessing behaviour. |
| `tests/test_rag.py` | Unit tests for RAG components. | Covers chunking, vector store, retriever, and prompt behaviour. |
| `requirements.txt` | Python dependencies. | Pinning compatible versions helps prevent ML-library conflicts. |
| `.env` | Local secrets/configuration. | Contains `GEMINI_API_KEY`; never commit it. |
| `QUICKSTART.md`, `BACKEND_DOCUMENTATION.md`, `RAG_DOCUMENTATION.md` | Existing project documentation. | Use them for setup and deeper component notes. |
| `temp.py`, `test_rag.html` | Local development/experiment files. | They are not required for the deployed application flow. |

`__init__.py` files mark Python packages. They do not contain core application logic.

## 5. ML prediction details you should be able to explain

The raw prediction inputs are:

- IPO date
- issue size
- QIB, HNI, and RII subscription values
- total subscription
- offer price

The code generates 16 features. Examples include:

- `year`, `month`, `quarter`, and `day_of_week` from the IPO date
- `QIB / Total`, `HNI / Total`, and `RII / Total`
- `log(Issue_Size)`
- `HNI - RII` and `QIB - RII`

The model returns probabilities for both classes. A positive-listing probability is a **model estimate**, not financial advice or a guarantee of investment returns.

## 6. RAG concepts you should know

- **Embedding:** a numeric vector representing semantic meaning. Similar text has vectors that are close together.
- **SentenceTransformer / MiniLM:** the local embedding model used for question and document vectors.
- **Chunking:** splitting a long document into smaller retrievable passages. This project preserves source filename and page metadata.
- **FAISS:** a vector-similarity-search library. It retrieves chunks whose embeddings are most similar to the question embedding.
- **RAG:** retrieve relevant context first, then give that context to an LLM to ground its answer.
- **Grounding:** the prompt tells Gemini to answer only from retrieved passages. This reduces hallucination but does not eliminate it completely.
- **Citations:** source filename/page data travels with each chunk and is returned to the UI.
- **Why not send the whole PDF to Gemini?** It is slow, expensive, less targeted, and can exceed context limits. Retrieval sends only relevant excerpts.

## 7. Likely interview questions and good answers

### Q1. What problem does IPOlytics solve?

It combines two IPO research tasks: a model-based listing prediction and fast question answering over a long prospectus. Users do not need to manually scan hundreds of PDF pages for common questions such as risks, business overview, or IPO proceeds.

### Q2. Why use FastAPI instead of putting everything in Streamlit?

FastAPI separates the UI from model and RAG logic. It provides typed APIs, reusable backend endpoints, easier testing, and allows the expensive resources to stay loaded in one backend process.

### Q3. Does Streamlit load the ML model or FAISS index?

No. Streamlit is only a client. It sends requests to FastAPI. The backend owns the joblib ML model, embedding model, FAISS index, and Gemini call.

### Q4. Explain the RAG flow in this project.

The backend embeds the user's question using MiniLM, searches the persisted FAISS index, selects relevant prospectus chunks, formats them in a grounded prompt, calls Gemini, and returns the answer with page citations.

### Q5. Does every question reprocess the prospectus?

No. PDF extraction, chunking, and document embeddings belong to offline ingestion. A normal query only embeds the short question and searches the saved FAISS index.

### Q6. What is stored in FAISS and what is stored separately?

FAISS stores the numeric embedding vectors in `index.faiss`. A pickle metadata file stores the chunk text, source filename, and page number because FAISS itself does not manage that application metadata.

### Q7. Why use normalized embeddings with `IndexFlatIP`?

When vectors are L2-normalized, inner product is equivalent to cosine similarity. This makes `IndexFlatIP` suitable for semantic similarity retrieval.

### Q8. Why use chunks instead of full pages or the complete PDF?

Chunks make retrieval more precise and keep the prompt small. A full PDF is too long and includes mostly irrelevant text for a specific question.

### Q9. What are QIB, HNI, and RII?

They are IPO investor categories: Qualified Institutional Buyers, High Net-worth Individuals/non-institutional investors, and Retail Individual Investors. Their subscription values are model inputs.

### Q10. Why is feature order important for the prediction model?

The model learned from features in a particular order. Supplying correctly calculated features in a different column order can produce invalid predictions, so `get_feature_names()` is used when building the inference DataFrame.

### Q11. How do you handle invalid input?

Pydantic validates API fields such as positive issue size, offer price, and total subscription. Feature engineering also rejects invalid dates and division by zero. The UI displays a friendly API error.

### Q12. What was the main performance bottleneck?

The external Gemini call, not FAISS. Retrieval was measured around 0.10 seconds, while Gemini depends on network latency and free-tier quotas.

### Q13. How do you handle Gemini failures?

The backend distinguishes rate-limit/quota errors and provider timeouts and returns helpful HTTP responses. The Streamlit UI shows a friendly error instead of a traceback.

### Q14. What are the project limitations?

It is a single-document portfolio demo. Gemini free-tier quotas can interrupt answers; RAG quality depends on text extraction/retrieval quality; and the prediction is an educational estimate, not investment advice.

### Q15. How would you improve it for production?

Add a controlled ingestion workflow, persistent document/version tracking, background jobs for indexing, authentication, observability, evaluation datasets, caching, stronger tests, and a paid/reliable LLM provider. I would add these only if product requirements justified the extra complexity.

### Q16. How would you evaluate RAG quality?

Create a small set of prospectus questions with expected source pages and answers. Measure retrieval hit rate, citation correctness, groundedness, answer completeness, latency, and failure rate.

### Q17. How do citations reduce hallucination?

They make the answer traceable to retrieved passages and encourage verification. They do not prove every claim is correct, so the prompt still instructs Gemini not to invent information.

### Q18. Why cache the embedding model and FAISS index?

Loading an embedding model and index on every request adds unnecessary latency. Loading them once at startup makes each question request lightweight.

## 8. Demo script for an interviewer

1. Start FastAPI: `python -m uvicorn app.main:app --reload`
2. Start Streamlit: `streamlit run frontend/streamlit_app.py`
3. Show **IPO Prediction**, enter values, and explain that the request goes to `/predict`.
4. Show **IPO Research**, ask: `What are the major risks mentioned in this IPO?`
5. Point to the answer and source pages.
6. Explain that the PDF was indexed beforehand and only the question embedding plus FAISS search runs during the demo.

## 9. Honest statements to use

- “This is a portfolio prototype built to demonstrate an end-to-end ML and RAG workflow.”
- “The Gemini free tier can return quota or deadline errors; the UI handles those cleanly.”
- “The prediction is educational and should not be treated as financial advice.”
- “The app uses one already indexed prospectus; multi-document ingestion is intentionally outside the scope of this version.”

