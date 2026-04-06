from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.state import PortfolioState, CriticOutput
from app.prompts.critic import CRITIC_PROMPT
import logging

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def critic_node(state: PortfolioState) -> dict:
    """
    Critic Agent — stress-tests the analyst's conclusions.
    Acts as a guardrail — if analysis fails, researcher is sent back.
    """
    logger.info(f"Critic evaluating {state['ticker']} (attempt {state['retry_count'] + 1})")

    research = state["research"]
    analysis = state["analysis"]

    structured_llm = llm.with_structured_output(CriticOutput)

    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=f"""
Stress-test this investment analysis:

=== RESEARCH DATA ===
Ticker: {research.ticker}
Data Confidence: {research.data_confidence:.0%}
Key Facts: {research.key_facts}
Missing/Uncertain Data points to check:
- P/E Ratio: {"Available" if research.pe_ratio else "MISSING"}
- Revenue Growth: {"Available" if research.revenue_growth else "MISSING"}
- Debt/Equity: {"Available" if research.debt_to_equity else "MISSING"}

=== ANALYST ASSESSMENT ===
Verdict: {analysis.analyst_verdict}
Overall Score: {analysis.overall_score}/10
Growth Score: {analysis.growth_score}/10
Risk Score: {analysis.risk_score}/10 (higher = riskier)
Valuation Score: {analysis.valuation_score}/10

Opportunities identified:
{chr(10).join(f"- {o}" for o in analysis.opportunities)}

Risks identified:
{chr(10).join(f"- {r}" for r in analysis.risks)}

=== GUARDRAIL CHECK ===
Apply these rules:
1. FAIL if data_confidence < 0.5
2. FAIL if overall_score > 7 AND risk_score > 7 (overconfident on risky asset)
3. FAIL if more than 2 critical metrics are missing (P/E, revenue growth, debt/equity)
4. FAIL if analyst provided fewer than 2 concrete risks

Current retry count: {state['retry_count']} (max 2 retries allowed)

Provide your CriticOutput with guardrail decision and revised risk assessment.
""")
    ]

    critic_output = structured_llm.invoke(messages)

    status = "PASSED" if critic_output.passes_guardrail else "❌ FAILED"
    logger.info(f"{status} guardrail — confidence: {critic_output.confidence_score:.0%}")

    return {
        "critique": critic_output,
        "critique_passed": critic_output.passes_guardrail,
        "retry_count": state["retry_count"] + (0 if critic_output.passes_guardrail else 1),
        "messages": [f"Critic: Guardrail {'PASSED' if critic_output.passes_guardrail else 'FAILED'} — {critic_output.critic_note[:100]}"]
    }