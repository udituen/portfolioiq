import yfinance as yf
from langchain.tools import tool
from typing import Dict, Any
import pandas as pd


@tool
def get_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Retrieve current stock price, market cap, and key financial ratios
    for a given ticker symbol using Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'TSLA')

    Returns:
        Dictionary with price, market cap, P/E ratio, revenue growth, debt/equity
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", "Unknown"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap": _format_market_cap(info.get("marketCap", 0)),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "profit_margin": info.get("profitMargins"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "analyst_target": info.get("targetMeanPrice"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "business_summary": info.get("longBusinessSummary", "")[:500],
        }
    except Exception as e:
        return {"error": f"Failed to retrieve data for {ticker}: {str(e)}"}


@tool
def get_stock_news(ticker: str) -> Dict[str, Any]:
    """
    Retrieve recent news headlines for a given stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')

    Returns:
        Dictionary with list of recent news headlines and links
    """
    try:
        stock = yf.Ticker(ticker.upper())
        news = stock.news[:5]  # latest 5 articles

        headlines = []
        for article in news:
            headlines.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher", ""),
                "link": article.get("link", ""),
            })

        return {
            "ticker": ticker.upper(),
            "news_count": len(headlines),
            "headlines": headlines
        }
    except Exception as e:
        return {"error": f"Failed to retrieve news for {ticker}: {str(e)}"}


@tool
def get_financial_history(ticker: str) -> Dict[str, Any]:
    """
    Retrieve 1-year price history and calculate key momentum metrics.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with price performance metrics
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="1y")

        if hist.empty:
            return {"error": f"No historical data for {ticker}"}

        current = hist["Close"].iloc[-1]
        year_ago = hist["Close"].iloc[0]
        month_ago = hist["Close"].iloc[-22] if len(hist) > 22 else hist["Close"].iloc[0]

        return {
            "ticker": ticker.upper(),
            "1y_return_pct": round(((current - year_ago) / year_ago) * 100, 2),
            "1m_return_pct": round(((current - month_ago) / month_ago) * 100, 2),
            "current_price": round(current, 2),
            "52w_high": round(hist["Close"].max(), 2),
            "52w_low": round(hist["Close"].min(), 2),
            "avg_volume_30d": int(hist["Volume"].tail(30).mean()),
        }
    except Exception as e:
        return {"error": f"Failed to retrieve history for {ticker}: {str(e)}"}


def _format_market_cap(market_cap: int) -> str:
    if not market_cap:
        return "Unknown"
    if market_cap >= 1_000_000_000_000:
        return f"${market_cap / 1_000_000_000_000:.1f}T"
    if market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.1f}B"
    if market_cap >= 1_000_000:
        return f"${market_cap / 1_000_000:.1f}M"
    return f"${market_cap:,}"