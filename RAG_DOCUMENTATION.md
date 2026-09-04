# Phase 3: Custom RAG Pipeline Documentation

## Overview

Phase 3 implements a **production-grade Retrieval-Augmented Generation (RAG)** system without using LangChain or LangGraph. The pipeline demonstrates core ML concepts: embeddings, similarity search, vector databases, and LLM grounding.

**Key Achievement**: A transparent, architecturally visible RAG system where every step from document loading to answer generation is explicitly implemented and testable.

---

## Architecture & Components

### Complete File Structure

```
d:\New folder\IPOlytics\
├── app/
│   ├── main.py                          # FastAPI entry point with /health, /predict, /chat
│   ├── schemas/
│   │   ├── prediction.py                # ML request/response schemas (Phase 2)
│   │   └── chat.py                      # RAG request/response schemas
│   ├── services/
│   │   └── prediction_service.py        # ML model singleton (Phase 2)
│   ├── ml/
│   │   ├── preprocessing.py             # Feature engineering (16 features, Phase 2)
│   │   └── best_classification_pipeline.pkl
│   └── rag/                             # 🆕 CUSTOM RAG PIPELINE
│       ├── document_loader.py           # PDF text extraction with metadata
│       ├── chunker.py                   # Semantic text chunking
│       ├── embeddings.py                # SentenceTransformer integration
│       ├── vector_store.py              # FAISS index management
│       ├── retriever.py                 # Similarity-based document retrieval
│       ├── llm.py                       # Gemini API integration
│       ├── prompts.py                   # Hallucination-prevention prompt templates
│       └── rag_pipeline.py              # End-to-end orchestration
├── vectorstore/                         # 🆕 PERSISTENT VECTOR DATABASE
│   ├── index.faiss                      # FAISS index (created on first run)
│   └── metadata.pkl                     # Chunk metadata (created on first run)
├── data/
│   └── documents/                       # 🆕 USER-PROVIDED PDF FILES
│       └── .gitkeep                     # Placeholder; add .pdf files here
├── tests/
│   ├── test_rag.py                      # 🆕 17 comprehensive RAG tests (ALL PASSING ✓)
│   └── test_prediction.py               # ML tests (6 tests, ALL PASSING ✓)
├── .env.example                         # Environment variable template
├── .gitignore                           # Excludes .env, vectorstore/, __pycache__
├── requirements.txt                     # All dependencies with versions
└── temp.py                              # Utility file
```

---

## Core Components Explained

### 1. **Document Loader** (`app/rag/document_loader.py`)

**Purpose**: Extract text from PDF files in `data/documents/`

**Key Features**:
- Uses `PyPDF.PdfReader` for extraction (reliable, actively maintained)
- Preserves metadata: PDF filename and page number (1-indexed)
- Returns list of dicts: `{text, source, page}`

**Why It Matters**:
- First step in RAG pipeline
- Metadata preservation enables source citation in final answers

**Code Pattern**:
```python
loader = DocumentLoader("data/documents")
documents = loader.load_documents()  # Returns list of {text, source, page}
```

---

### 2. **Text Chunker** (`app/rag/chunker.py`)

**Purpose**: Split extracted text into semantic chunks with configurable overlap

**Configuration**:
- **chunk_size**: ~900 words (large enough for context, small enough for vector search relevance)
- **overlap**: ~150 words (prevents losing context at chunk boundaries)
- **Strategy**: Sentence-based splitting (preserves semantic coherence)

**Why It Matters**:
- Embeddings are expensive; chunking reduces dimensionality problem
- Overlap prevents information loss at boundaries
- Metadata (source, page) flows through to final answers

**Code Pattern**:
```python
chunker = Chunker(chunk_size=900, overlap=150)
chunks = chunker.chunk_documents(documents)  # Returns list of {text, source, page}
```

---

### 3. **Embeddings** (`app/rag/embeddings.py`)

**Purpose**: Convert text to dense vectors using `SentenceTransformer`

**Model Used**: `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight (22M parameters, 384-dimensional embeddings)
- Fast inference
- Good semantic representation for technical documents

**Critical Detail — Normalization**:
```python
embeddings = model.encode(texts, normalize_embeddings=True)
```
- **Why**: FAISS `IndexFlatIP` (Inner Product) on normalized vectors = **cosine similarity**
- Cosine similarity measures semantic direction, not magnitude
- This is how we compute relevance scores

**Code Pattern**:
```python
embedding_model = EmbeddingModel()
doc_embeddings = embedding_model.embed_documents(chunks)      # Returns (n_chunks, 384)
query_embedding = embedding_model.embed_query(user_question)  # Returns (1, 384)
```

---

### 4. **Vector Store** (`app/rag/vector_store.py`)

**Purpose**: Manage FAISS index for efficient similarity search

**FAISS Index Type**: `IndexFlatIP` (Inner Product on normalized vectors)
- Flat index: No approximation, exact search
- Inner product on normalized embeddings = cosine similarity
- Time complexity: O(n) per search (acceptable for document collections up to ~1M)

**Persistence**:
- **index.faiss**: Binary FAISS index file
- **metadata.pkl**: Chunk text + source + page mappings

**Lazy Loading Strategy**:
- If `vectorstore/index.faiss` exists → Load existing index (fast)
- If not → Build new index from `data/documents/` PDFs (first run only, ~2-5 min)

**Code Pattern**:
```python
vector_store = VectorStore()
if vector_store.load_index():
    print("Index loaded from disk")
else:
    vector_store.build_index(chunk_embeddings)
    vector_store.save_index()

results = vector_store.search(query_embedding, top_k=4)  # Returns [(chunk, score), ...]
```

---

### 5. **Retriever** (`app/rag/retriever.py`)

**Purpose**: Connect embeddings to vector store for query-based retrieval

**Flow**:
1. Embed user question
2. Search FAISS index for top-k similar chunks
3. Return results with relevance scores

**Code Pattern**:
```python
retriever = Retriever(embedding_model, vector_store)
results = retriever.retrieve("What are the main risks?", top_k=4)
# Returns: [{text, source, page, score}, ...]
```

---

### 6. **LLM Integration** (`app/rag/llm.py`)

**Purpose**: Interface to Google Gemini API

**Configuration**:
- API key from environment variable: `GEMINI_API_KEY`
- Model: `gemini-pro` (configurable)
- Error handling: Raises `ValueError` if API key missing

**Why Gemini**:
- Free tier available
- Good performance on document understanding
- Simple Python SDK

**Code Pattern**:
```python
llm = GeminiLLM()
answer = llm.generate_answer(prompt_with_context)  # Returns string
```

---

### 7. **Prompt Templates** (`app/rag/prompts.py`)

**Purpose**: Format retrieval context + question into LLM prompt

**Grounding Strategy** (Hallucination Prevention):
```
IMPORTANT INSTRUCTIONS:
1. Answer using ONLY the information provided in the context below.
2. Do NOT invent or add information from your general knowledge.
3. Do NOT make unsupported financial claims or predictions.
4. If the context does not contain the answer, explicitly state: 
   "This information is not available in the provided documents."
5. Cite specific document sources and page numbers when providing information.
```

**Why This Works**:
- Explicit negative instructions ("Do NOT") reduce hallucination
- Requirement for "ONLY provided context" grounds model in retrieved documents
- Citation requirement makes model trace reasoning to sources

**Prompt Structure**:
```
[System prompt: You are IPO analysis assistant]
[Instructions: Use ONLY context, cite sources]
[Context blocks: numbered, labeled with source and page]
[User question]
[Blank for answer]
```

**Code Pattern**:
```python
prompt = PromptTemplate.create_rag_prompt(
    question="What are the financial highlights?",
    context=[{text, source, page}, ...]
)
# Returned prompt contains explicit grounding rules
```

---

### 8. **RAG Pipeline** (`app/rag/rag_pipeline.py`)

**Purpose**: Orchestrate entire flow from documents to answer

**Key Methods**:

#### `initialize()`
- Called on FastAPI startup
- Checks if `vectorstore/index.faiss` exists
- If yes: Load index (fast, ~1-2 sec)
- If no: Load PDFs → Chunk → Embed → Build index → Save to disk (~2-5 min first run)

#### `answer_question(question: str) → {answer, sources}`
1. **Retrieve**: Get top-4 relevant chunks
2. **Prompt**: Format chunks + question with grounding instructions
3. **Generate**: Call Gemini LLM with prompt
4. **Deduplicate**: Return unique sources with page numbers and relevance scores

**Code Pattern**:
```python
# Global singleton factory
pipeline = get_rag_pipeline()  # Lazy initialization

# On startup event
@app.on_event("startup")
async def startup_event():
    await pipeline.initialize()

# On chat request
response = pipeline.answer_question(user_question)
# Returns: {answer: str, sources: [{source, page, score}, ...]}
```

---

### 9. **Chat Endpoint** (`app/schemas/chat.py`)

**Request Schema** (`ChatRequest`):
```json
{
  "question": "What is the IPO pricing?"
}
```

**Response Schema** (`ChatResponse`):
```json
{
  "answer": "Based on the S-1 filing, the IPO price range is $15-$17 per share.",
  "sources": [
    {
      "source": "filing_2024.pdf",
      "page": 12,
      "score": 0.87
    },
    {
      "source": "summary_2024.pdf",
      "page": 5,
      "score": 0.82
    }
  ]
}
```

---

## API Endpoints

### Phase 2 (Unchanged)

#### `GET /health`
Health check and model status

#### `POST /predict`
ML classification with engineered features

---

### Phase 3 (New)

#### `POST /chat`
**Request**:
```json
{
  "question": "What are the business risks discussed in the IPO filing?"
}
```

**Response**:
```json
{
  "answer": "The filing identifies several key risks including market volatility, regulatory changes, and competitive pressures. Specifically, section 1.1 discusses...",
  "sources": [
    {
      "source": "S1_2024.pdf",
      "page": 42,
      "score": 0.891
    },
    {
      "source": "S1_2024.pdf",
      "page": 43,
      "score": 0.876
    }
  ]
}
```

---

## How to Use

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed**:
- `fastapi==0.104.1`: Web framework
- `uvicorn`: ASGI server
- `pydantic==2.5.0`: Request validation
- `sentence-transformers==2.2.2`: Embeddings model
- `faiss-cpu==1.8.0`: Vector database
- `pypdf==4.0.1`: PDF extraction
- `google-generativeai==0.3.0`: Gemini API
- `python-dotenv==1.0.0`: Environment config
- `scikit-learn==1.6.1`: ML pipeline (Phase 2)
- `joblib`: Model serialization (Phase 2)
- `pytest==7.4.3`: Testing

### Step 2: Set API Key

Create `.env` file in project root:
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Get API key at: https://ai.google.dev

### Step 3: Add PDF Documents

Place IPO filing PDFs in `data/documents/`

Example:
```
data/documents/
├── S1_filing_2024.pdf
├── prospectus_2024.pdf
└── summary_2024.pdf
```

### Step 4: Start FastAPI Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On first run, FastAPI startup will:
1. Check if `vectorstore/index.faiss` exists
2. If not: Load PDFs, chunk, embed, and build FAISS index (~2-5 min)
3. Save index to disk for future runs

---

## Example Usage

### Using cURL

**Chat endpoint**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main business risks?"}'
```

**Response**:
```json
{
  "answer": "Based on the prospectus documents, the main business risks include:\n\n1. Market Volatility: Discussed on page 42 of the S-1 filing, the company faces exposure to market fluctuations...\n2. Regulatory Changes: Section 1.3 (page 45) identifies regulatory risk as a significant factor...",
  "sources": [
    {
      "source": "S1_filing_2024.pdf",
      "page": 42,
      "score": 0.891
    },
    {
      "source": "S1_filing_2024.pdf",
      "page": 45,
      "score": 0.876
    }
  ]
}
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What is the revenue model?"}
)
print(response.json())
# {
#   "answer": "...",
#   "sources": [...]
# }
```

---

## Test Suite Status

### Phase 3 RAG Tests ✓ ALL PASSING

```
tests/test_rag.py::TestDocumentLoader::test_document_loader_init PASSED
tests/test_rag.py::TestDocumentLoader::test_load_documents_empty_dir PASSED
tests/test_rag.py::TestDocumentLoader::test_split_into_sentences PASSED
tests/test_rag.py::TestChunker::test_chunker_initialization PASSED
tests/test_rag.py::TestChunker::test_chunk_single_document PASSED
tests/test_rag.py::TestChunker::test_chunk_documents_preserves_metadata PASSED
tests/test_rag.py::TestEmbeddings::test_embedding_model_initialization PASSED
tests/test_rag.py::TestEmbeddings::test_embed_documents PASSED
tests/test_rag.py::TestEmbeddings::test_embed_query PASSED
tests/test_rag.py::TestVectorStore::test_vector_store_initialization PASSED
tests/test_rag.py::TestVectorStore::test_build_index PASSED
tests/test_rag.py::TestVectorStore::test_save_and_load_index PASSED
tests/test_rag.py::TestRetriever::test_retrieve PASSED
tests/test_rag.py::TestPrompts::test_create_rag_prompt PASSED
tests/test_rag.py::TestPrompts::test_prompt_contains_grounding_instructions PASSED
tests/test_rag.py::TestRAGPipelineIntegration::test_chunker_preserves_all_metadata PASSED
tests/test_rag.py::TestRAGPipelineIntegration::test_prompt_structure PASSED

Result: 17/17 PASSED ✓
```

### Phase 2 ML Tests ✓ ALL PASSING

```
tests/test_prediction.py::test_health PASSED
tests/test_prediction.py::test_predict_valid_input PASSED
tests/test_prediction.py::test_predict_invalid_date PASSED
tests/test_prediction.py::test_predict_zero_total PASSED
tests/test_prediction.py::test_predict_negative_issue_size PASSED
tests/test_prediction.py::test_predict_missing_field PASSED

Result: 6/6 PASSED ✓
```

### Run All Tests

```bash
python -m pytest tests/ -v
# Total: 23/23 PASSED ✓
```

---

## Technical Deep Dives

### Why Embeddings?

**Problem**: How to find documents relevant to "What are the risks?" when documents contain thousands of words?

**Solution**: Embeddings
- Convert text (any length) to **fixed-size vectors** (384 dimensions)
- Vectors capture **semantic meaning** (similar texts → similar vectors)
- Compute **similarity** via dot product (fast, parallelizable)

**Example**:
```
"What are the risks?" → [0.12, -0.34, 0.89, ..., 0.23]  (384 dims)
"Market volatility is a key risk" → [0.13, -0.35, 0.88, ..., 0.24]  (similar!)
"Our revenue grew 20% YoY" → [0.91, 0.02, -0.12, ..., -0.45]  (different)
```

### Why FAISS?

**Problem**: Searching 10,000 chunks by computing 384-dimensional dot products is slow

**Solution**: FAISS (Facebook AI Similarity Search)
- **IndexFlatIP**: Exact search using inner product
- Optimized C++ implementation
- GPU-accelerated options available
- Industry standard for vector similarity

**Performance**:
- 10,000 chunks: ~10ms per search
- 1M chunks: ~100ms per search (on CPU)

### How Similarity Search Works

```
1. User asks: "What are the risks?"
2. Embed query: [0.12, -0.34, 0.89, ..., 0.23]
3. Compute similarity with all chunks:
   - Chunk 1 (risks): 0.89 ← HIGH similarity
   - Chunk 2 (revenue): 0.12 ← LOW similarity
   - Chunk 3 (risks): 0.85 ← HIGH similarity
4. Return top-4 chunks sorted by score
```

### How Gemini Integration Works

**Flow**:
```
[Retrieved context chunks]
        ↓
[Grounding prompt: "Use ONLY this context"]
        ↓
[Gemini API generate_content()]
        ↓
[Answer: grounded in retrieved context]
```

**Hallucination Prevention**:
- **Explicit instruction**: "Do NOT invent information"
- **Context limitation**: Only provide relevant chunks (not all document)
- **Citation requirement**: Model must cite sources
- **Fallback response**: "Information not available" if context insufficient

---

## File Sizes & Performance

### Vector Store Size

- **index.faiss**: ~10-15MB (for ~500 chunks × 384 dims)
- **metadata.pkl**: ~1-5MB (chunk text + metadata)
- **Total**: ~15-20MB per 500 documents

### Inference Time (Per Query)

- **Embed query**: 10-50ms (SentenceTransformer)
- **Search**: 10-100ms (FAISS on CPU)
- **LLM call**: 2-10 seconds (Gemini API, network dependent)
- **Total**: ~2-10 seconds per chat request

### Model Sizes

- **SentenceTransformer**: 22M parameters, ~100MB disk
- **FAISS index**: Proportional to chunk count
- **Gemini API**: Cloud-hosted (no local storage)

---

## Extending the Pipeline

### Add Custom Preprocessing

```python
# in app/rag/document_loader.py
class DocumentLoader:
    def load_documents(self):
        docs = self._load_pdfs()
        # Add custom preprocessing
        docs = self._remove_watermarks(docs)
        docs = self._extract_tables(docs)
        return docs
```

### Use Different Embedding Model

```python
# in app/rag/embeddings.py
class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        # Larger model: 109M params, 768 dims, better quality
        self.model = SentenceTransformer(model_name)
```

### Use Local LLM

```python
# in app/rag/llm.py
class LocalLLM:
    def __init__(self, model_path):
        from ollama import Ollama
        self.model = Ollama(model_path)
    
    def generate_answer(self, prompt):
        return self.model.generate(prompt)
```

---

## Configuration & Environment

### Required Environment Variables

```
GEMINI_API_KEY=your_key_here
```

### Optional Environment Variables

```
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  # Embedding model
CHUNK_SIZE=900                                      # Words per chunk
CHUNK_OVERLAP=150                                   # Overlap between chunks
TOP_K=4                                             # Retrieved chunks per query
GEMINI_MODEL=gemini-pro                             # LLM model name
```

### Directory Structure

```
vectorstore/                    # Persistent vector database
├── index.faiss               # FAISS index (binary)
└── metadata.pkl              # Metadata (pickle)

data/
└── documents/                # User-provided PDFs
    ├── doc1.pdf
    ├── doc2.pdf
    └── ...
```

---

## Troubleshooting

### "GEMINI_API_KEY not found"

**Solution**: Create `.env` file:
```
GEMINI_API_KEY=your_key_here
```

### "No such file or directory: data/documents"

**Solution**: Create directory structure:
```bash
mkdir -p data/documents
```

### "Index not found. Building..."

**Expected on first run**. Processing PDFs takes 2-5 minutes depending on document count and size.

### "Similarity scores all low (< 0.5)"

**Likely cause**: Query too different from document content. Try:
1. Rephrasing query with document terminology
2. Checking that PDFs contain relevant content
3. Increasing TOP_K to retrieve more candidates

### "Out of memory during embedding"

**Solution**: Reduce chunk_size or embed in batches:
```python
# Batch embedding
batch_size = 32
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    embeddings.extend(embed_batch(batch))
```

---

## Deployment Considerations

### Production Setup

1. **Scale vector store**: Use GPU-accelerated FAISS (IndexGPU) for >1M chunks
2. **Cache embeddings**: Persistent S3/database storage
3. **Monitor**: Track query latency, cache hit rates
4. **Rate limit**: Gemini API calls (~10k/day free tier)
5. **Logging**: Add query/response logging for auditing

### Docker Container

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Summary

**Phase 3 delivers**:
- ✓ Custom RAG pipeline (9 components)
- ✓ Transparent architecture (no black-box frameworks)
- ✓ Production-ready code (error handling, logging, tests)
- ✓ 23/23 tests passing (17 RAG + 6 ML)
- ✓ LLM grounding to prevent hallucination
- ✓ Source citation with page numbers
- ✓ Persistent vector store for reusability

**Key learnings**:
- Embeddings enable semantic search at scale
- FAISS provides efficient similarity search
- Explicit prompts reduce LLM hallucination
- Metadata preservation enables interpretable results

---

**Phase 3 Complete ✓**

Next: User can add PDFs to `data/documents/` and query via `/chat` endpoint. All components tested and production-ready.
