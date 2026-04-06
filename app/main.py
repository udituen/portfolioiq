from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PortfolioIQ starting up...")
    from app.tools.vector_store import get_vector_store
    get_vector_store()  # Pre-warm FAISS index
    logger.info("Vector store initialised")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="PortfolioIQ — Multi-Agent Investment Analysis",
    description="4-agent LangGraph workflow for investment research and decision making",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class AnalysisRequest(BaseModel):
    ticker: str
    query: str = "Should I invest in this stock?"


class AnalysisResponse(BaseModel):
    ticker: str
    recommendation: str
    confidence: float
    target_price_range: str
    time_horizon: str
    key_factors: list
    key_risks: list
    rationale: str
    disclaimer: str
    agent_log: list
    research_summary: dict
    analyst_scores: dict
    critic_passed: bool



@app.get("/")
def root():
    return {
        "service": "PortfolioIQ",
        "version": "1.0.0",
        "description": "Multi-agent investment analysis powered by LangGraph",
        "docs": "/docs",
        "endpoints": {
            "analyse": "POST /analyse",
            "health": "GET /health"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "langsmith_configured": bool(os.getenv("LANGCHAIN_API_KEY")),
    }


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(request: AnalysisRequest):
    """
    Run the full 4-agent PortfolioIQ workflow for a given ticker.

    The agents run in sequence:
    Researcher → Analyst → Critic → Decision

    The Critic may route back to the Researcher if data quality is insufficient.
    """
    ticker = request.ticker.upper().strip()

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

    logger.info(f"Starting analysis for {ticker}")

    try:
        from app.graph import portfolio_graph

        # Initial state
        initial_state = {
            "ticker": ticker,
            "query": request.query,
            "research": None,
            "analysis": None,
            "critique": None,
            "decision": None,
            "messages": [],
            "research_complete": False,
            "analysis_complete": False,
            "critique_passed": False,
            "retry_count": 0,
        }

        # Run the graph
        final_state = portfolio_graph.invoke(initial_state)

        decision = final_state.get("decision")
        research = final_state.get("research")
        analysis = final_state.get("analysis")
        critique = final_state.get("critique")

        if not decision:
            raise HTTPException(status_code=500, detail="Agent workflow did not produce a decision")

        return AnalysisResponse(
            ticker=ticker,
            recommendation=decision.recommendation,
            confidence=decision.confidence,
            target_price_range=decision.target_price_range,
            time_horizon=decision.time_horizon,
            key_factors=decision.key_factors,
            key_risks=decision.key_risks,
            rationale=decision.rationale,
            disclaimer=decision.disclaimer,
            agent_log=final_state.get("messages", []),
            research_summary={
                "current_price": research.current_price if research else None,
                "market_cap": research.market_cap if research else None,
                "data_confidence": research.data_confidence if research else None,
            },
            analyst_scores={
                "overall": analysis.overall_score if analysis else None,
                "growth": analysis.growth_score if analysis else None,
                "risk": analysis.risk_score if analysis else None,
                "valuation": analysis.valuation_score if analysis else None,
                "verdict": analysis.analyst_verdict if analysis else None,
            },
            critic_passed=critique.passes_guardrail if critique else False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")