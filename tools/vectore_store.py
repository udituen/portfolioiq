import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_core.documents import Document


# ── Global FAISS store ─────────────────────────────────────────
_vector_store: Optional[FAISS] = None
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Seed knowledge — financial concepts the agent can reference
SEED_DOCUMENTS = [
    Document(
        page_content="P/E ratio (Price-to-Earnings) measures how much investors pay per dollar of earnings. A P/E below 15 is generally considered undervalued, 15-25 is fair value, above 25 may be overvalued. Compare to industry average for context.",
        metadata={"source": "financial_concepts", "topic": "valuation"}
    ),
    Document(
        page_content="Debt-to-Equity ratio measures financial leverage. Below 1.0 indicates conservative financing. Above 2.0 may indicate high financial risk. Technology companies often carry lower D/E ratios than capital-intensive industries.",
        metadata={"source": "financial_concepts", "topic": "risk"}
    ),
    Document(
        page_content="Revenue growth rate indicates business momentum. Growth above 15% annually is considered strong. Declining revenue growth is a warning signal. Compare to industry peers and overall market growth.",
        metadata={"source": "financial_concepts", "topic": "growth"}
    ),
    Document(
        page_content="Profit margin measures operational efficiency. Net profit margin above 20% is excellent. Below 5% may indicate pricing pressure or high costs. Margins expanding over time signal improving business quality.",
        metadata={"source": "financial_concepts", "topic": "profitability"}
    ),
    Document(
        page_content="52-week price range provides context on current valuation. A stock trading near its 52-week low may be undervalued or facing fundamental challenges. Near 52-week high may indicate momentum or overvaluation.",
        metadata={"source": "financial_concepts", "topic": "price_analysis"}
    ),
    Document(
        page_content="Forward P/E uses projected earnings estimates. A lower forward P/E than trailing P/E suggests expected earnings growth. Significant difference may indicate analyst optimism — verify with revenue growth trends.",
        metadata={"source": "financial_concepts", "topic": "valuation"}
    ),
    Document(
        page_content="Investment risk assessment framework: Market risk (broad market decline), Company risk (business-specific issues), Liquidity risk (inability to sell at fair price), Regulatory risk (policy changes affecting the business). All four should be evaluated.",
        metadata={"source": "financial_concepts", "topic": "risk_framework"}
    ),
    Document(
        page_content="BUY recommendation criteria: Strong earnings growth, reasonable valuation (P/E below sector average), manageable debt, positive revenue trend, clear competitive advantage. SELL criteria: deteriorating margins, rising debt, losing market share, valuation stretched.",
        metadata={"source": "decision_framework", "topic": "recommendations"}
    ),
]


def get_vector_store() -> FAISS:
    """Get or initialise the FAISS vector store with seed knowledge."""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISS.from_documents(SEED_DOCUMENTS, embeddings)
    return _vector_store


def add_to_store(documents: List[Document]) -> int:
    """Add new documents to the vector store."""
    store = get_vector_store()
    store.add_documents(documents)
    return len(documents)


@tool
def search_financial_knowledge(query: str) -> str:
    """
    Search the financial knowledge base for relevant concepts,
    frameworks, and benchmarks to inform investment analysis.

    Args:
        query: Financial concept or question to search for

    Returns:
        Relevant financial knowledge and benchmarks
    """
    try:
        store = get_vector_store()
        docs = store.similarity_search(query, k=3)

        if not docs:
            return "No relevant financial knowledge found for this query."

        results = []
        for doc in docs:
            results.append(f"[{doc.metadata.get('topic', 'general')}] {doc.page_content}")

        return "\n\n".join(results)
    except Exception as e:
        return f"Knowledge base search failed: {str(e)}"