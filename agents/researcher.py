from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.state import PortfolioState, ResearchOutput
from app.prompts.researcher import RESEARCHER_PROMPT
from app.tools.yahoo_finance import get_stock_data, get_stock_news, get_financial_history
from app.tools.vector_store import search_financial_knowledge
import json
import logging

logger = logging.getLogger(__name__)

# Tools available to the researcher
RESEARCHER_TOOLS = [get_stock_data, get_stock_news, get_financial_history, search_financial_knowledge]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
researcher_llm = llm.bind_tools(RESEARCHER_TOOLS)


def researcher_node(state: PortfolioState) -> dict:
    """
    Research Agent — gathers factual market data using Yahoo Finance tools.
    Populates state with ResearchOutput.
    """
    logger.info(f"🔍 Researcher starting for {state['ticker']}")

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"""
Research the following stock for investment analysis:

Ticker: {state['ticker']}
User Query: {state['query']}

Use your tools to gather:
1. Current price, market cap, and financial ratios (get_stock_data)
2. Recent news headlines (get_stock_news)
3. 1-year price performance (get_financial_history)
4. Relevant financial benchmarks (search_financial_knowledge)

Then provide a structured ResearchOutput with all findings.
Assess your data confidence honestly (0-1 score).
""")
    ]

    # Agentic loop — keep calling tools until research complete
    max_iterations = 4
    for i in range(max_iterations):
        response = researcher_llm.invoke(messages)
        messages.append(response)

        # If no more tool calls — extract structured output
        if not response.tool_calls:
            break

        # Execute tool calls
        from langchain_core.messages import ToolMessage
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Route to correct tool
            tool_map = {
                "get_stock_data": get_stock_data,
                "get_stock_news": get_stock_news,
                "get_financial_history": get_financial_history,
                "search_financial_knowledge": search_financial_knowledge,
            }

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                messages.append(ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                    content=str(result)
                ))

    # Extract structured ResearchOutput from final response
    structured_llm = llm.with_structured_output(ResearchOutput)
    research_messages = [
        SystemMessage(content="Extract the research findings into a structured ResearchOutput. Be precise with numbers."),
        HumanMessage(content=f"Based on this research conversation, extract the structured output:\n\n{str(messages[-1].content)}")
    ]
    research_output = structured_llm.invoke(research_messages)

    logger.info(f"✅ Research complete for {state['ticker']} — confidence: {research_output.data_confidence}")

    return {
        "research": research_output,
        "research_complete": True,
        "messages": [f"Researcher: Gathered data for {state['ticker']} with {research_output.data_confidence:.0%} confidence"]
    }