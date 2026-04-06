from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.state import PortfolioState, DecisionOutput
from app.prompts.decision import DECISION_PROMPT
import logging

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def decision_node(state: PortfolioState) -> dict:
    """
    Decision Agent — synthesises all prior agent outputs into a final recommendation.
    Produces structured DecisionOutput with BUY/HOLD/SELL/INSUFFICIENT_DATA.
    """
    logger.info(f"Decision Agent synthesising for {state['ticker']}")

    research = state["research"]
    analysis = state["analysis"]
    critique = state["critique"]

    structured_llm = llm.with_structured_output(DecisionOutput)

    messages = [
        SystemMessage(content=DECISION_PROMPT),
        HumanMessage(content=f"""
Synthesise all agent perspectives into a final investment recommendation:

=== RESEARCH SUMMARY ===
Ticker: {research.ticker} | {research.company_name}
Price: ${research.current_price} | Market Cap: {research.market_cap}
P/E: {research.pe_ratio} | Revenue Growth: {research.revenue_growth} | D/E: {research.debt_to_equity}
Data Confidence: {research.data_confidence:.0%}

=== ANALYST VIEW ===
Verdict: {analysis.analyst_verdict} | Score: {analysis.overall_score}/10
Growth: {analysis.growth_score}/10 | Risk: {analysis.risk_score}/10 | Valuation: {analysis.valuation_score}/10
Top Opportunities: {analysis.opportunities[:3]}
Top Risks: {analysis.risks[:3]}

=== CRITIC ASSESSMENT ===
Guardrail: {"PASSED" if critique.passes_guardrail else "FAILED ❌"}
Confidence: {critique.confidence_score:.0%}
Revised Risk Score: {critique.revised_risk_score}/10
Key Challenges: {critique.challenges[:3]}
Missing Data: {critique.missing_data}
Critic Note: {critique.critic_note}

=== AGENT MESSAGE LOG ===
{chr(10).join(state['messages'])}

=== USER QUERY ===
{state['query']}

Retries used: {state['retry_count']}

Produce your final DecisionOutput. Be decisive. Be balanced. Include the disclaimer.
""")
    ]

    decision_output = structured_llm.invoke(messages)

    logger.info(f"Decision: {decision_output.recommendation} (confidence: {decision_output.confidence:.0%})")

    return {
        "decision": decision_output,
        "messages": [f"Decision: {decision_output.recommendation} — {decision_output.rationale[:100]}"]
    }