# IPOlytics — AI-Powered IPO Listing Prediction & Research Assistant

IPOlytics is an end-to-end AI application that combines **machine learning-based IPO listing prediction** with a **RAG-powered research assistant**.

## Features

- **IPO Listing Prediction** using Scikit-learn
- **RAG-based IPO Research Assistant**
- **FAISS vector similarity search**
- **MiniLM embeddings**
- **Gemini API** for grounded answer generation
- **FastAPI REST APIs**
- **Streamlit frontend**
- **Source/page references** for RAG answers
- Automated tests using **Pytest**

## Architecture

```text
                    ┌──────────────────┐
                    │   Streamlit UI   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    └───────┬───┬──────┘
                            │   │
                    /predict│   │/chat
                            │   │
                            ▼   ▼
                     ┌───────┐ ┌─────────────┐
                     │  ML   │ │ RAG Pipeline│
                     │Model  │ └──────┬──────┘
                     └───────┘        │
                                      ▼
                                Query Embedding
                                      │
                                      ▼
                                  FAISS Search
                                      │
                                      ▼
                                Relevant Chunks
                                      │
                                      ▼
                                  Gemini API
                                      │
                                      ▼
                              Grounded Answer
