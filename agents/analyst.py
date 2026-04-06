from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.state import PortfolioState, AnalystOutput
from app.prompts.analyst import ANALYST_PROMPT
from app.tools.vector_store import search_financial_knowledge
import logging

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


def analyst_node(state: PortfolioState) -> dict:
    """
    Analyst Agent — interprets research data and scores the investment opportunity.
    Uses structured output to produce AnalystOutput.
    """
    logger.info(f"Analyst starting for {state['ticker']}")

    research = state["research"]

    # Fetch relevant benchmarks from knowledge base
    benchmarks = search_financial_knowledge.invoke(
        {"query": f"investment scoring {research.ticker} {research.company_name}"}
    )

    structured_llm = llm.with_structured_output(AnalystOutput)

    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=f"""
Analyse this investment opportunity:

=== RESEARCH DATA ===
Ticker: {research.ticker}
Company: {research.company_name}
Current Price: ${research.current_price}
Market Cap: {research.market_cap}
P/E Ratio: {research.pe_ratio}
Revenue Growth: {research.revenue_growth}
Debt/Equity: {research.debt_to_equity}

Key Facts:
{chr(10).join(f"- {fact}" for fact in research.key_facts)}

Recent News:
{chr(10).join(f"- {news}" for news in research.recent_news)}

Data Confidence: {research.data_confidence:.0%}

=== FINANCIAL BENCHMARKS ===
{benchmarks}

=== ORIGINAL QUERY ===
{state['query']}

Produce your AnalystOutput with scores (0-10), opportunities, risks, and verdict.
""")
    ]

    analysis_output = structured_llm.invoke(messages)

    logger.info(f"Analysis complete — verdict: {analysis_output.analyst_verdict}, score: {analysis_output.overall_score:.1f}/10")

    return {
        "analysis": analysis_output,
        "analysis_complete": True,
        "messages": [f"Analyst: {analysis_output.analyst_verdict} — overall score {analysis_output.overall_score:.1f}/10"]
    }