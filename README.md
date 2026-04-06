# portfolioiq (In Development)
A 4-agent LangGraph workflow (Researcher → Analyst → Critic → Decision) with conditional edge routing, RAG-powered financial data retrieval (Yahoo Finance, FAISS), and a structured Critic guardrail layer; deployed via FastAPI and Docker to GCP Cloud Run with LangSmith tracing for agent observability.


## Architecture

```
User Query (ticker + question)
         ↓
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Researcher  │───▶│   Analyst    │───▶│    Critic   │   │
│  │              │    │              │    │  (Guardrail) │   │
│  │ Yahoo Finance│    │ Score 0-10   │    │ PASS / FAIL  │   │
│  │ FAISS RAG    │    │ BULL/BEAR    │◀───│ (retry loop) │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘   │
│                                                 │ PASS      │
│                                          ┌──────▼───────┐   │
│                                          │   Decision   │   │
│                                          │ BUY/HOLD/SELL│   │
│                                          └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ↓
Structured JSON recommendation with confidence score,
target price range, key factors, risks, and rationale
```

---

## Agent Roles

| Agent | Role | Tools |
|---|---|---|
| **Researcher** | Gathers factual market data — no interpretation | Yahoo Finance (price, news, history) + FAISS knowledge base |
| **Analyst** | Scores the investment opportunity (0-10) | Structured output via `with_structured_output` |
| **Critic** | Stress-tests analysis — guardrail PASS/FAIL | Routes back to Researcher if data insufficient |
| **Decision** | Final BUY/HOLD/SELL synthesis | Weighs all three prior agent outputs |

---


## Tech Stack

Technology 
|---|
LangGraph 1.0 
 OpenAI GPT-4o-mini 
Yahoo Finance (yfinance) 
FAISS + OpenAI Embeddings 
FastAPI + Pydantic 
LangSmith tracing 
Docker 
GCP Cloud Run 
GitHub Actions + Workload Identity Federation 

---

<!-- ## Running Locally

```bash
# Clone and setup
git clone https://github.com/udituen/portfolioiq
cd portfolioiq

# Environment
cp .env.example .env
# Add OPENAI_API_KEY and LANGCHAIN_API_KEY to .env

# Install and run
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080

# Visit http://localhost:8080/docs
```

### Run with Docker

```bash
docker-compose up --build
```

### Run Tests

```bash
pytest tests/ -v
```

---

## LangSmith Observability

Every agent run is automatically traced in LangSmith when `LANGCHAIN_TRACING_V2=true`. You can see:
- Each agent's inputs and outputs
- Tool calls made by the Researcher
- Routing decisions at the Critic node
- Full token usage and latency per agent

Sign up free at **smith.langchain.com**

---

## Author

**Uduak Ituen** — AI Engineer | MLOps | LangGraph

[LinkedIn](https://linkedin.com/in/uduak-ituen) · [GitHub](https://github.com/udituen) · [Diabetes MLOps API](https://diabetes-api-565467329909.us-central1.run.app/docs) -->
