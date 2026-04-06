from langgraph.graph import StateGraph, START, END
from app.state import PortfolioState
from app.agents.researcher import researcher_node
from app.agents.analyst import analyst_node
from app.agents.critic import critic_node
from app.agents.decision import decision_node
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ── Conditional edge functions ─────────────────────────────────

def route_after_critic(state: PortfolioState) -> str:
    """
    Conditional routing after the Critic node.

    Routes:
    - "decision"    → critique passed, proceed to final recommendation
    - "researcher"  → critique failed AND retries remaining, go back for more data
    - "decision"    → critique failed BUT max retries reached, force decision with INSUFFICIENT_DATA note
    """
    if state["critique_passed"]:
        logger.info("✅ Routing to Decision Agent")
        return "decision"

    if state["retry_count"] <= MAX_RETRIES:
        logger.info(f"🔄 Routing back to Researcher (retry {state['retry_count']}/{MAX_RETRIES})")
        return "researcher"

    # Max retries exhausted — force decision with caveat
    logger.warning("⚠️ Max retries reached — routing to Decision with insufficient data flag")
    return "decision"


def route_after_research(state: PortfolioState) -> str:
    """
    After research, always proceed to analyst.
    Could be extended to short-circuit if ticker is invalid.
    """
    if state.get("research") and state["research"].data_confidence > 0:
        return "analyst"
    return END


# ── Graph assembly ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble the PortfolioIQ multi-agent graph.

    Flow:
    START → researcher → analyst → critic →
        [passes]  → decision → END
        [fails, retries left] → researcher → analyst → critic → ...
        [fails, max retries]  → decision → END
    """
    graph = StateGraph(PortfolioState)

    # Add nodes
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("decision", decision_node)

    # Entry point
    graph.add_edge(START, "researcher")

    # Researcher → Analyst (always, if research succeeded)
    graph.add_conditional_edges(
        "researcher",
        route_after_research,
        {"analyst": "analyst", END: END}
    )

    # Analyst → Critic (always)
    graph.add_edge("analyst", "critic")

    # Critic → Decision or back to Researcher (conditional)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "decision": "decision",
            "researcher": "researcher",
        }
    )

    # Decision → END
    graph.add_edge("decision", END)

    return graph.compile()


# Compiled graph — singleton
portfolio_graph = build_graph()