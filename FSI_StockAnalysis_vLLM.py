#!/usr/bin/env python3
"""
AMD Instinct MI3xx ROCm-Powered Financial Analysis Demo - vLLM Version
Converted from Ollama to vLLM for enhanced performance on AMD GPU infrastructure
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import time
import re
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

# vLLM API Configuration
VLLM_API_BASE = "http://localhost:8003/v1"  # Using LiteLLM proxy endpoint
MODEL_NAME = "microsoft/phi-4"  # Your configured model
MAX_TOKENS = 2048
TEMPERATURE = 0.3

def call_vllm_api(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
    """
    Call vLLM API through LiteLLM proxy
    
    Args:
        prompt: The input prompt for the model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text response
    """
    try:
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        response = requests.post(
            f"{VLLM_API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return f"Error: Failed to get response from vLLM API (Status: {response.status_code})"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may be processing a complex query."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to vLLM API. Please ensure the service is running."
    except Exception as e:
        return f"Error: {str(e)}"

def get_stock_data(symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical stock data for given symbols.
    
    Args:
        symbols: List of stock symbols to fetch
        period: Period of data to fetch (e.g., '1y', '6mo', '3mo')
        interval: Data interval (e.g., '1d', '1h')
        
    Returns:
        Dictionary with symbol as key and DataFrame as value
    """
    stock_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(period=period, interval=interval)
            if not data.empty:
                stock_data[symbol.upper()] = data
                print(f"✅ Successfully fetched data for {symbol.upper()}: {len(data)} data points")
            else:
                print(f"❌ No data found for {symbol.upper()}")
        except Exception as e:
            print(f"❌ Error fetching data for {symbol.upper()}: {str(e)}")
    
    return stock_data

def get_stock_data_date_range(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical stock data for given symbols within a specific date range.
    
    Args:
        symbols: List of stock symbols to fetch
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary with symbol as key and DataFrame as value
    """
    stock_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(start=start_date, end=end_date)
            if not data.empty:
                stock_data[symbol.upper()] = data
                print(f"✅ Successfully fetched data for {symbol.upper()}: {len(data)} data points from {start_date} to {end_date}")
            else:
                print(f"❌ No data found for {symbol.upper()} in the specified date range")
        except Exception as e:
            print(f"❌ Error fetching data for {symbol.upper()}: {str(e)}")
    
    return stock_data

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various technical indicators for stock analysis.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    try:
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_Change_1D'] = df['Close'].pct_change(1)
        df['Price_Change_5D'] = df['Close'].pct_change(5)
        df['Price_Change_20D'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_20D'] = df['Close'].rolling(window=20).std()
        
        return df
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        return df

def get_company_info(symbol: str) -> Dict:
    """
    Get company information and recent news.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with company info
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        company_data = {
            'symbol': symbol,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'price_to_book': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'recommendation': info.get('recommendationKey', 'N/A'),
            'target_price': info.get('targetMeanPrice', 0)
        }
        
        return company_data
    except Exception as e:
        print(f"Error getting company info for {symbol}: {str(e)}")
        return {'symbol': symbol, 'name': 'N/A', 'error': str(e)}

def get_recent_headlines(symbol: str, limit: int = 5) -> List[str]:
    """
    Get recent news headlines for a stock.
    
    Args:
        symbol: Stock symbol
        limit: Number of headlines to return
        
    Returns:
        List of recent headlines
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        headlines = []
        for article in news[:limit]:
            headline = article.get('title', '')
            if headline:
                headlines.append(headline)
        
        return headlines
    except Exception as e:
        print(f"Error getting headlines for {symbol}: {str(e)}")
        return [f"Error retrieving news for {symbol}"]

def create_stock_summary(symbol: str, df: pd.DataFrame, company_info: Dict) -> str:
    """
    Create a comprehensive stock summary for LLM analysis.
    
    Args:
        symbol: Stock symbol
        df: DataFrame with stock data and technical indicators
        company_info: Company information dictionary
        
    Returns:
        Formatted summary string
    """
    try:
        latest_data = df.iloc[-1]
        latest_close = latest_data['Close']
        
        # Calculate recent performance
        performance_1d = ((latest_data['Close'] / df.iloc[-2]['Close']) - 1) * 100
        performance_1w = ((latest_data['Close'] / df.iloc[-5]['Close']) - 1) * 100 if len(df) >= 5 else 0
        performance_1m = ((latest_data['Close'] / df.iloc[-21]['Close']) - 1) * 100 if len(df) >= 21 else 0
        
        # Technical levels
        sma_20 = latest_data.get('SMA_20', 0)
        sma_50 = latest_data.get('SMA_50', 0)
        rsi = latest_data.get('RSI', 0)
        
        # Recent headlines
        headlines = get_recent_headlines(symbol, 3)
        headlines_text = "\n".join([f"• {headline}" for headline in headlines])
        
        summary = f"""
STOCK ANALYSIS DATA FOR {symbol} ({company_info.get('name', 'N/A')}):

COMPANY OVERVIEW:
• Sector: {company_info.get('sector', 'N/A')}
• Industry: {company_info.get('industry', 'N/A')}
• Market Cap: ${company_info.get('market_cap', 0):,.0f}
• P/E Ratio: {company_info.get('pe_ratio', 0):.2f}
• Beta: {company_info.get('beta', 0):.2f}

CURRENT PRICE ACTION:
• Current Price: ${latest_close:.2f}
• 1-Day Change: {performance_1d:.2f}%
• 1-Week Change: {performance_1w:.2f}%
• 1-Month Change: {performance_1m:.2f}%
• 52-Week High: ${company_info.get('fifty_two_week_high', 0):.2f}
• 52-Week Low: ${company_info.get('fifty_two_week_low', 0):.2f}

TECHNICAL INDICATORS:
• 20-Day SMA: ${sma_20:.2f}
• 50-Day SMA: ${sma_50:.2f}
• RSI (14): {rsi:.2f}
• Position vs SMA20: {'Above' if latest_close > sma_20 else 'Below'} (+{((latest_close/sma_20-1)*100):.2f}%)
• Position vs SMA50: {'Above' if latest_close > sma_50 else 'Below'} (+{((latest_close/sma_50-1)*100):.2f}%)

RECENT NEWS HEADLINES:
{headlines_text}

TRADING METRICS:
• Average Volume (20-day): {df['Volume'].rolling(20).mean().iloc[-1]:,.0f}
• Latest Volume: {latest_data['Volume']:,.0f}
• Volume vs Average: {(latest_data['Volume'] / df['Volume'].rolling(20).mean().iloc[-1]):.2f}x
• 20-Day Volatility: {latest_data.get('Volatility_20D', 0):.2f}%
"""
        return summary
    except Exception as e:
        return f"Error creating summary for {symbol}: {str(e)}"

def generate_analysis_prompt(symbols: List[str], stock_summaries: List[str], investor_type: str) -> str:
    """
    Generate the analysis prompt for the LLM.
    
    Args:
        symbols: List of stock symbols
        stock_summaries: List of stock summary strings
        investor_type: Type of investor (Conservative, Moderate, Aggressive, Day Trader)
        
    Returns:
        Formatted prompt string
    """
    
    combined_data = "\n" + "="*80 + "\n".join(stock_summaries)
    
    investor_profiles = {
        "Conservative": "low-risk, dividend-focused, long-term holdings with stable companies",
        "Moderate": "balanced risk-reward, mix of growth and value stocks, medium-term investments",
        "Aggressive": "high-risk high-reward, growth stocks, emerging markets, higher volatility tolerance",
        "Day Trader": "short-term technical trading, momentum plays, intraday movements, quick profits"
    }
    
    profile_description = investor_profiles.get(investor_type, investor_profiles["Moderate"])
    
    prompt = f"""
You are an expert financial analyst with decades of experience in equity research and technical analysis. You have been asked to provide a comprehensive stock analysis for a {investor_type.lower()} investor profile who seeks {profile_description}.

ANALYSIS REQUEST:
Analyze the following stock(s): {', '.join(symbols)}

STOCK DATA:
{combined_data}

ANALYSIS REQUIREMENTS:

1. TECHNICAL ANALYSIS SUMMARY:
   - Analyze key technical indicators (SMA trends, RSI levels, volume patterns)
   - Identify support and resistance levels
   - Comment on momentum and trend direction
   - Note any technical patterns or signals

2. FUNDAMENTAL OVERVIEW:
   - Evaluate valuation metrics (P/E, P/B ratios)
   - Assess market position and competitive advantages
   - Consider sector and industry trends
   - Review recent news impact

3. MARKET SENTIMENT & NEWS ANALYSIS:
   - Interpret recent headlines and their potential impact
   - Assess overall market sentiment for each stock
   - Identify key catalysts or risk factors

4. RISK ASSESSMENT:
   - Evaluate volatility and beta risk
   - Identify company-specific risks
   - Consider sector and market risks
   - Assess liquidity and trading volume

5. INVESTOR PROFILE ALIGNMENT:
   For a {investor_type} investor seeking {profile_description}:
   - Rate each stock's suitability (1-10 scale)
   - Explain alignment with investment goals
   - Suggest position sizing considerations
   - Recommend holding period

6. ACTIONABLE INSIGHTS:
   - Provide clear technical entry/exit levels
   - Suggest portfolio allocation percentages
   - Identify key metrics to monitor
   - Set realistic price targets

Please provide a thorough, professional analysis that balances technical precision with practical investment guidance. Use clear, actionable language suitable for an informed investor.

Focus on delivering specific, quantitative insights rather than generic advice. Include concrete price levels, percentages, and timeframes where appropriate.
"""
    
    return prompt

def extract_recommendations(analysis_text: str, symbols: List[str]) -> str:
    """
    Extract clear buy/sell/hold recommendations from the analysis.
    
    Args:
        analysis_text: Full analysis text from LLM
        symbols: List of stock symbols analyzed
        
    Returns:
        Formatted recommendations summary
    """
    
    # Create a prompt to extract recommendations
    extraction_prompt = f"""
Based on the following stock analysis, provide clear, concise buy/sell/hold recommendations for each stock: {', '.join(symbols)}

ANALYSIS TEXT:
{analysis_text}

Please extract and format the recommendations as follows:

For each stock, provide:
1. RECOMMENDATION: Buy/Sell/Hold
2. CONFIDENCE LEVEL: High/Medium/Low
3. KEY REASONING: 2-3 bullet points
4. PRICE TARGET: If mentioned
5. TIME HORIZON: Short/Medium/Long term
6. POSITION SIZE: Suggested allocation if mentioned

Format as:
**[SYMBOL]**: [RECOMMENDATION] ([CONFIDENCE])
• Reason 1
• Reason 2
• Target: $X.XX | Horizon: [TIME] | Size: [%]

Keep it concise and actionable. Focus on the most important factors driving each recommendation.
"""
    
    # Call vLLM API to extract recommendations
    recommendations = call_vllm_api(extraction_prompt, max_tokens=1500, temperature=0.1)
    return recommendations

def create_stock_chart(symbol: str, df: pd.DataFrame) -> plt.Figure:
    """
    Create a comprehensive stock chart with technical indicators.
    
    Args:
        symbol: Stock symbol
        df: DataFrame with stock data and technical indicators
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), 
                                           gridspec_kw={'height_ratios': [3, 1, 1]},
                                           facecolor='white')
        
        # Main price chart
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='#2E86AB')
        ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7, color='#A23B72')
        ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7, color='#F18F01')
        
        # Bollinger Bands
        if 'BB_Upper' in df.columns:
            ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.2, color='gray', label='Bollinger Bands')
        
        ax1.set_title(f'{symbol} - Stock Price Analysis', fontsize=16, fontweight='bold', color='#2c3e50')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        colors = ['green' if close >= open_ else 'red' for close, open_ in zip(df['Close'], df['Open'])]
        ax2.bar(df.index, df['Volume'], alpha=0.6, color=colors)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_title('Trading Volume', fontsize=12, color='#2c3e50')
        ax2.grid(True, alpha=0.3)
        
        # RSI chart
        if 'RSI' in df.columns:
            ax3.plot(df.index, df['RSI'], color='purple', linewidth=2)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax3.fill_between(df.index, 70, 100, alpha=0.1, color='red')
            ax3.fill_between(df.index, 0, 30, alpha=0.1, color='green')
            ax3.set_ylabel('RSI', fontsize=12)
            ax3.set_title('Relative Strength Index (RSI)', fontsize=12, color='#2c3e50')
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error creating chart for {symbol}: {str(e)}")
        # Return a simple error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error creating chart for {symbol}\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Chart Error - {symbol}')
        return fig

def format_number(num: Union[int, float]) -> str:
    """Format numbers for display."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def gradio_interface(symbols_input: str, start_date: str, end_date: str, investor_type: str) -> Tuple:
    """
    Main Gradio interface function that processes user input and returns analysis results.
    
    Args:
        symbols_input: Comma-separated stock symbols
        start_date: Start date for analysis
        end_date: End date for analysis
        investor_type: Type of investor profile
        
    Returns:
        Tuple containing analysis results for Gradio outputs
    """
    
    # Record start time for performance metrics
    start_time = time.time()
    
    try:
        # Parse symbols
        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        if not symbols:
            return "Please enter at least one stock symbol.", "", None, "0", "0", "0"
        
        # Validate dates
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD format.", "", None, "0", "0", "0"
        
        print(f"🚀 Starting analysis for: {', '.join(symbols)}")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"👤 Investor type: {investor_type}")
        
        # Fetch stock data
        stock_data = get_stock_data_date_range(symbols, start_date, end_date)
        
        if not stock_data:
            return "No data found for the specified symbols and date range.", "", None, "0", "0", "0"
        
        # Process each stock
        stock_summaries = []
        total_data_points = 0
        
        for symbol in symbols:
            if symbol in stock_data:
                df = stock_data[symbol].copy()
                
                # Calculate technical indicators
                df = calculate_technical_indicators(df)
                
                # Get company info
                company_info = get_company_info(symbol)
                
                # Create summary
                summary = create_stock_summary(symbol, df, company_info)
                stock_summaries.append(summary)
                
                total_data_points += len(df)
                print(f"✅ Processed {symbol}: {len(df)} data points")
        
        if not stock_summaries:
            return "Unable to process any of the specified symbols.", "", None, "0", "0", "0"
        
        # Generate analysis prompt
        analysis_prompt = generate_analysis_prompt(symbols, stock_summaries, investor_type)
        
        # Get LLM analysis using vLLM
        print("🤖 Generating AI analysis with vLLM...")
        ai_analysis = call_vllm_api(analysis_prompt, max_tokens=2048, temperature=0.3)
        
        if ai_analysis.startswith("Error:"):
            return ai_analysis, "", None, "0", "0", "0"
        
        # Extract recommendations
        print("💡 Extracting recommendations...")
        recommendations = extract_recommendations(ai_analysis, symbols)
        
        # Create chart for first symbol
        chart_fig = None
        if symbols and symbols[0] in stock_data:
            print(f"📊 Creating chart for {symbols[0]}...")
            chart_fig = create_stock_chart(symbols[0], stock_data[symbols[0]])
        
        # Calculate performance metrics
        end_time = time.time()
        inference_time = f"{end_time - start_time:.2f}"
        
        # Estimate token count (rough approximation)
        estimated_tokens = len(analysis_prompt.split()) + len(ai_analysis.split())
        token_count = format_number(estimated_tokens)
        
        data_points = format_number(total_data_points)
        
        print(f"✅ Analysis completed in {inference_time} seconds")
        print(f"📊 Processed {data_points} data points")
        
        return (
            ai_analysis,
            recommendations,
            chart_fig,
            inference_time,
            token_count,
            data_points
        )
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, "", None, "0", "0", "0"

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for professional AMD styling
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global font improvements */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Monospace for code/data */
    code, .gr-textbox textarea, .gr-number input {
        font-family: 'JetBrains Mono', 'SF Mono', Monaco, Consolas, monospace !important;
        font-weight: 500 !important;
    }
    
    /* Professional headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }
    
    /* AMD brand colors and professional styling */
    .gradio-container {
        font-size: 14px !important;
        line-height: 1.6 !important;
        color: #1a1a1a !important;
    }
    
    /* Professional button styling */
    .gr-button-primary {
        background: linear-gradient(135deg, #ED1C24 0%, #C5282F 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.02em !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(237, 28, 36, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(237, 28, 36, 0.4) !important;
    }
    
    /* Professional text inputs */
    .gr-textbox label, .gr-dropdown label, .gr-number label {
        font-weight: 600 !important;
        font-size: 13px !important;
        color: #2c3e50 !important;
        letter-spacing: 0.01em !important;
    }
    
    /* Hide the default dropdown button for tabs */
    .gradio-tabs button[data-tab-id]:not([data-tab-id*="__tab"]) ~ button:last-child {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Force all tab buttons to be visible */
    .gradio-tabs .tab-nav button {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Ensure all tabs are visible and container uses full width */
    .gradio-tabs .tab-nav {
        height: auto !important;
        max-height: none !important;
        width: 100% !important;
    }
    
    /* Professional card styling */
    .gr-box {
        border-radius: 12px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Enhanced markdown styling */
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        font-weight: 700 !important;
        color: #1a1a1a !important;
    }
    
    .gr-markdown p {
        font-size: 14px !important;
        line-height: 1.7 !important;
        color: #4a5568 !important;
    }
    
    .gr-markdown strong {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    """
    
    with gr.Blocks(title="AMD Instinct MI3xx vLLM-Powered Financial Analysis Demo", theme=gr.themes.Soft(), css=custom_css) as interface:
        # Professional AMD header with enhanced typography
        gr.HTML("""
            <div style="position: relative; padding: 24px 28px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; margin-bottom: 24px; border: 1px solid rgba(0,0,0,0.06);">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" alt="AMD Logo" style="position: absolute; top: 20px; right: 28px; height: 40px; width: auto;" />
                <div style="padding-right: 140px;">
                    <h1 style="margin: 0; color: #1a1a1a; font-size: 2.4em; font-weight: 800; font-family: 'Inter', sans-serif; letter-spacing: -0.03em; line-height: 1.1;">AMD Instinct vLLM Financial Analysis</h1>
                    <h3 style="margin: 8px 0 0 0; color: #ED1C24; font-size: 1.1em; font-weight: 600; font-family: 'Inter', sans-serif; letter-spacing: 0.01em;">Powered by vLLM + ROCm Platform</h3>
                    <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9em; font-weight: 400;">Enterprise-Grade GPU-Accelerated AI Analysis</p>
                </div>
            </div>
        """)
        
        gr.Markdown("""
            **Next-Generation AI Stock Analysis with vLLM on AMD Hardware**
            
            This advanced financial analysis tool leverages **vLLM** for high-performance LLM inference on AMD's Instinct GPU 
            architecture, delivering lightning-fast AI-driven stock analysis with enterprise-grade GPU acceleration.

            ### 🎯 Key Enhancements:
            - **vLLM Engine**: Optimized LLM serving with higher throughput
            - **AMD ROCm Integration**: Native GPU acceleration for AI workloads  
            - **LiteLLM Proxy**: Unified API interface for multiple models
            - **Real-time Analysis**: Sub-second inference times with GPU optimization
            - **Enterprise Scale**: Production-ready deployment architecture

        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Configuration")
                
                symbols_input = gr.Textbox(
                    label="Stock Symbol(s) (e.g., AAPL, MSFT)", 
                    placeholder="Enter one or more stock symbols, separated by commas...",
                    lines=2
                )
                
                # Date selection with calendar pickers
                gr.Markdown("#### 📅 Date Selection")
                with gr.Row():
                    with gr.Column():
                        start_date_input = gr.Textbox(
                            label="Start Date (YYYY-MM-DD)",
                            value="2024-08-13",
                            placeholder="2024-01-01",
                            info="Enter start date in YYYY-MM-DD format"
                        )

                    with gr.Column():
                        end_date_input = gr.Textbox(
                            label="End Date (YYYY-MM-DD)",
                            value="2025-01-13",
                            placeholder="2025-01-01",
                            info="Enter end date in YYYY-MM-DD format"
                        )
                
                investor_type_input = gr.Dropdown(
                    choices=["Conservative", "Moderate", "Aggressive", "Day Trader"],
                    label="Investor Type",
                    value="Moderate",
                    info="Select your investment profile for personalized recommendations"
                )
                
                analyze_btn = gr.Button("🚀 Analyze with vLLM", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analysis Results")
                
                with gr.Tabs():
                    with gr.TabItem("🤖 vLLM Technical Analysis"):
                        ai_analysis_output = gr.Textbox(
                            label="vLLM AI Technical & Market Analysis", 
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.TabItem("💡 Buy/Sell/Hold Recommendations"):
                        recommendations_output = gr.Textbox(
                            label="Investment Recommendations",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.TabItem("📊 Stock Charts & Price Analysis"):
                        chart_output = gr.Plot(label="Stock Price Chart (First Symbol)")
        
        # Performance Metrics
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚡ vLLM Performance Metrics")
                with gr.Row():
                    inference_time_output = gr.Textbox(label="vLLM Inference Time (s)", interactive=False, scale=1)
                    token_count_output = gr.Textbox(label="Token Count", interactive=False, scale=1)
                    data_points_output = gr.Textbox(label="Data Points Analyzed", interactive=False, scale=1)
        
        # Add JavaScript for horizontal tabs
        gr.HTML("""
        <script>
        function ensureTabsVisible() {
            const tabContainers = document.querySelectorAll('.gradio-tabs');
            tabContainers.forEach(container => {
                const tabNav = container.querySelector('div:first-child');
                if (tabNav) {
                    tabNav.style.display = 'flex';
                    tabNav.style.flexWrap = 'nowrap';
                    tabNav.style.gap = '8px';
                    tabNav.style.width = '100%';
                    
                    const buttons = tabNav.querySelectorAll('button');
                    buttons.forEach(btn => {
                        if (!btn.textContent.includes('...') && btn.textContent.trim() !== '') {
                            btn.style.display = 'block';
                            btn.style.visibility = 'visible';
                            btn.style.flex = '1';
                        } else {
                            btn.style.display = 'none';
                        }
                    });
                }
            });
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(ensureTabsVisible, 500);
            setTimeout(ensureTabsVisible, 1500);
        });
        </script>
        """)
        
        # Enhanced styling for vLLM demo
        gr.HTML("""
        <style>
        .gradio-tabs .tab-nav,
        .gradio-tabs > div:first-child {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            background: #f8f9fa !important;
            padding: 12px !important;
            border-radius: 8px !important;
            margin-bottom: 10px !important;
            gap: 8px !important;
            border: 1px solid #e0e0e0 !important;
            min-height: 50px !important;
        }
        
        .gradio-tabs .tab-nav button,
        .gradio-tabs > div:first-child > button {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            flex: 1 !important;
            min-width: 150px !important;
            max-width: 250px !important;
            height: auto !important;
            padding: 8px 12px !important;
            background: white !important;
            border: 2px solid #e0e0e0 !important;
            border-radius: 6px !important;
            color: #333 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            text-align: center !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        
        .gradio-tabs .tab-nav button:hover,
        .gradio-tabs > div:first-child > button:hover {
            border-color: #00C2DE !important;
            background: rgba(0, 194, 222, 0.1) !important;
            color: #00C2DE !important;
        }
        
        .gradio-tabs .tab-nav button.selected,
        .gradio-tabs > div:first-child > button.selected,
        .gradio-tabs .tab-nav button[aria-selected="true"],
        .gradio-tabs > div:first-child > button[aria-selected="true"] {
            background: #00C2DE !important;
            border-color: #00C2DE !important;
            color: white !important;
        }
        
        .gradio-tabs button[aria-label*="More"],
        .gradio-tabs .tab-nav-button {
            display: none !important;
        }
        
        .gradio-tabs > div:nth-child(2) {
            background: transparent !important;
            border: none !important;
            margin-top: 10px !important;
        }
        </style>
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=gradio_interface,
            inputs=[symbols_input, start_date_input, end_date_input, investor_type_input],
            outputs=[
                ai_analysis_output,
                recommendations_output,
                chart_output,
                inference_time_output,
                token_count_output,
                data_points_output
            ]
        )
        
        # Example section with vLLM specific information
        gr.Markdown("""
            ### 💡 vLLM Demo Usage
            1. **Enter Stock Symbols**: Add symbols like "AAPL, MSFT, GOOGL" for analysis
            2. **Set Date Range**: Choose your analysis period
            3. **Select Profile**: Pick investment style for tailored recommendations  
            4. **Analyze**: Click to start vLLM-powered GPU-accelerated analysis
            5. **Review Results**: Explore technical analysis, recommendations, and charts

            ### 🚀 vLLM Advantages
            - **Higher Throughput**: 2-5x faster inference than standard serving
            - **Memory Efficiency**: Optimized GPU memory usage with PagedAttention
            - **Scalable**: Production-ready deployment architecture
            - **AMD ROCm Native**: Full GPU acceleration on AMD Instinct hardware

            ### ⚠️ Disclaimer
            This tool demonstrates vLLM capabilities for educational purposes. Always conduct independent research and consult financial advisors before investment decisions.
        """)
    
    return interface

# Create and launch interface
if __name__ == "__main__":
    print("🚀 Starting AMD Instinct vLLM-Powered Financial Analysis Demo...")
    print(f"🔗 vLLM API Endpoint: {VLLM_API_BASE}")
    print(f"🤖 Model: {MODEL_NAME}")
    
    iface = create_interface()
    iface.launch(
        server_name="0.0.0.0", 
        server_port=7861,
        share=True,
        debug=True
    )
