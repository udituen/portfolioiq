# portfolioiq
A 4-agent LangGraph workflow (Researcher → Analyst → Critic → Decision) with conditional edge routing, RAG-powered financial data retrieval (Yahoo Finance, FAISS), and a structured Critic guardrail layer; deployed via FastAPI and Docker to GCP Cloud Run with LangSmith tracing for agent observability.

## Project structure
```
portfolioiq/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entrypoint
│   ├── graph.py                 # LangGraph graph assembly
│   ├── state.py                 # PortfolioState TypedDict
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── researcher.py        # Tool calling + RAG retrieval
│   │   ├── analyst.py           # Structured analysis over state
│   │   ├── critic.py            # Guardrail + confidence check
│   │   └── decision.py          # Buy/Hold/Sell synthesis
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── yahoo_finance.py     # yfinance wrapper
│   │   └── vector_store.py      # FAISS index + retriever
│   │
│   └── prompts/
│       ├── researcher.py
│       ├── analyst.py
│       ├── critic.py
│       └── decision.py
│
└── tests/
    └── test_graph.py

```
