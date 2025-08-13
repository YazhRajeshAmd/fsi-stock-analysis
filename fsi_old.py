import yfinance as yf
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import re
from datetime import datetime, timedelta
import time
import requests

# Initialize Ollama with Llama3.1 - increased temperature for more varied responses
llm = Ollama(model="llama3.1:70b", temperature=0.3)

# Updated prompt template with momentum, price vs SMA, and investor type
stock_analysis_prompt = PromptTemplate(
    input_variables=["stock_data", "stock_symbol", "start_date", "end_date", "start_price", "end_price", "sma", "rsi", "news_headlines", "momentum", "price_vs_sma", "investor_type"],
    template="""
You are a financial analyst AI. Analyze the following stock data and provide insights tailored to the investor profile:

Stock Symbol: {stock_symbol}
Date Range: {start_date} to {end_date}
Starting Price: ${start_price}
Ending Price: ${end_price}
Investor Type: {investor_type}

Stock Data:
{stock_data}

Technical Indicators:
SMA (20): {sma}
RSI (14): {rsi}
Price Momentum: {momentum}%
Price vs SMA: {price_vs_sma}

Recent News Headlines:
{news_headlines}

IMPORTANT: Tailor your analysis and recommendation specifically for a {investor_type}:

Conservative Investor Profile:
- Prioritizes capital preservation and steady income
- Prefers established companies with strong fundamentals
- Low risk tolerance, seeks dividend-paying stocks
- Recommendation criteria: Strong balance sheet, consistent earnings, low volatility

Moderate Investor Profile:
- Balanced approach between growth and stability
- Willing to accept moderate risk for better returns
- Diversified portfolio with mix of growth and value stocks
- Recommendation criteria: Good growth potential with reasonable risk

Aggressive Investor Profile:
- High risk tolerance, seeks maximum capital appreciation
- Comfortable with volatility and market fluctuations
- Focuses on growth stocks and emerging opportunities
- Recommendation criteria: High growth potential, innovative companies

Day Trader Profile:
- Short-term trading focus (minutes to days)
- Technical analysis driven decisions
- High risk tolerance with quick profit/loss realization
- Recommendation criteria: High volume, volatility, clear technical patterns

Please provide a comprehensive analysis including:
1. Overall trend of the stock price
2. Key statistics (average price, highest price, lowest price)
3. Technical indicator interpretation (SMA, RSI) specific to {investor_type}
4. Any notable events or patterns observed
5. Volume analysis and its correlation with price changes
6. Impact of recent news on the stock
7. Risk assessment appropriate for {investor_type}
8. Investment horizon considerations for {investor_type}
9. A brief outlook for the stock based on this historical data

Based on your analysis and the {investor_type} profile, provide a clear recommendation: BUY, SELL, or HOLD.

Use these guidelines adjusted for {investor_type}:

Conservative Investor:
- BUY: Stable upward trend, price above SMA, RSI 30-60, positive fundamentals, dividend yield
- SELL: Declining trend with fundamental concerns, high volatility, RSI > 75
- HOLD: Stable but uncertain outlook, maintain existing positions

Moderate Investor:
- BUY: Upward trend (momentum > 3%), price above SMA, RSI < 70, balanced risk/reward
- SELL: Downward trend (momentum < -3%), price below SMA, RSI > 75, negative outlook
- HOLD: Mixed signals, sideways trend (-3% <= momentum <= 3%)

Aggressive Investor:
- BUY: Strong momentum (> 5%), growth potential, breaking resistance levels
- SELL: Severe downtrend (< -10%), breaking support levels, negative growth prospects
- HOLD: Consolidation phase, waiting for breakout signals

Day Trader:
- BUY: Strong intraday momentum, high volume, clear technical breakout
- SELL: Reversal patterns, profit-taking levels reached, volume decline
- HOLD: Consolidation, low volume, unclear technical signals

At the end of your analysis, clearly state: "RECOMMENDATION FOR {investor_type}: [BUY/SELL/HOLD]"

Make sure your final recommendation is one of: BUY, SELL, or HOLD.

Analysis:
"""
)

# Create an LLMChain for stock analysis
stock_analysis_chain = LLMChain(llm=llm, prompt=stock_analysis_prompt, verbose=True)

def get_stock_data(symbol, start_date, end_date):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        end_adjusted = end + timedelta(days=1)
        stock = yf.Ticker(symbol)
        data = stock.history(start=start, end=end_adjusted)
        data = data.loc[start_date:end_date]
        return data
    except Exception as e:
        return pd.DataFrame()

def get_technical_indicators(data):
    indicators = {}
    
    # Simple Moving Average (20)
    indicators['sma'] = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else None
    
    # Relative Strength Index (14)
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    indicators['rsi'] = rsi.iloc[-1] if len(rsi) >= 14 else None
    
    # Price momentum (% change over period)
    if len(data) > 1:
        indicators['momentum'] = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    else:
        indicators['momentum'] = 0
    
    # Current price vs SMA signal
    if indicators['sma'] is not None:
        indicators['price_vs_sma'] = "ABOVE" if data['Close'].iloc[-1] > indicators['sma'] else "BELOW"
    else:
        indicators['price_vs_sma'] = "N/A"
    
    return indicators

# Enhanced News API integration with multiple sources
def get_news_headlines(symbol, max_headlines=5):
    headlines = []
    
    # Method 1: Try Yahoo Finance RSS
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item')[:max_headlines]:
                title_elem = item.find('title')
                if title_elem is not None and title_elem.text:
                    headlines.append(f"- {title_elem.text}")
    except Exception as e:
        print(f"Yahoo RSS failed: {e}")
    
    # Method 2: Try Yahoo Finance ticker info for recent news
    if not headlines:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if news:
                for article in news[:max_headlines]:
                    if 'title' in article:
                        headlines.append(f"- {article['title']}")
        except Exception as e:
            print(f"Yahoo ticker news failed: {e}")
    
    # Method 3: Fallback - generate generic market context
    if not headlines:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            headlines = [
                f"- {company_name} operates in the {sector} sector",
                f"- Company industry: {industry}",
                f"- Recent market activity for {symbol} should be monitored",
                f"- General market sentiment may impact {sector} stocks",
                f"- Technical analysis recommended for {symbol} trading decisions"
            ]
        except Exception as e:
            print(f"Fallback info failed: {e}")
            headlines = [
                f"- No recent news headlines available for {symbol}",
                f"- Consider checking financial news websites for latest updates",
                f"- Market analysis based on technical indicators recommended",
                f"- General market conditions may affect stock performance"
            ]
    
    return '\n'.join(headlines[:max_headlines])

# Alternative: Simple market context function
def get_market_context(symbol):
    """Provide general market context when news is unavailable"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        context_items = []
        
        # Company basics
        if 'longName' in info:
            context_items.append(f"- Company: {info['longName']}")
        
        if 'sector' in info:
            context_items.append(f"- Sector: {info['sector']}")
            
        if 'industry' in info:
            context_items.append(f"- Industry: {info['industry']}")
        
        # Market cap and size context
        if 'marketCap' in info and info['marketCap']:
            market_cap = info['marketCap']
            if market_cap > 200_000_000_000:
                context_items.append("- Large-cap stock with established market presence")
            elif market_cap > 10_000_000_000:
                context_items.append("- Mid-cap stock with growth potential")
            else:
                context_items.append("- Small-cap stock with higher volatility potential")
        
        # Performance context
        if 'recommendationKey' in info:
            context_items.append(f"- Analyst recommendation: {info['recommendationKey']}")
        
        return '\n'.join(context_items) if context_items else f"- General market analysis for {symbol}"
        
    except Exception:
        return f"- Technical analysis recommended for {symbol}"

def plot_stock_data(data, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.fill_between(data.index, data['Low'], data['High'], alpha=0.2, color='lightblue')
    # Technical indicators
    if len(data) >= 20:
        sma = data['Close'].rolling(window=20).mean()
        plt.plot(data.index, sma, label='SMA (20)', color='orange')
    if len(data) >= 14:
        # RSI is not plotted on price chart, but could be shown in a subplot
        pass
    plt.title(f'{symbol} Stock Price', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt

def extract_recommendation(analysis):
    # Try multiple patterns to catch the recommendation
    patterns = [
        r"RECOMMENDATION FOR [^:]+:\s*(BUY|SELL|HOLD)",
        r"RECOMMENDATION:\s*(BUY|SELL|HOLD)",
        r"(BUY|SELL|HOLD)\s*(?:recommendation|decision|action)",
        r"My recommendation.*?is\s*(BUY|SELL|HOLD)",
        r"I recommend.*?(BUY|SELL|HOLD)",
        r"Final.*?recommendation.*?(BUY|SELL|HOLD)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, analysis, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()
    
    # If no explicit recommendation found, look for strong indicators
    if re.search(r"(strong\s+buy|definitely\s+buy|recommend\s+buying)", analysis, re.IGNORECASE):
        return "BUY"
    elif re.search(r"(strong\s+sell|definitely\s+sell|recommend\s+selling)", analysis, re.IGNORECASE):
        return "SELL"
    elif re.search(r"(hold|maintain|keep|stay)", analysis, re.IGNORECASE):
        return "HOLD"
    
    return "UNCLEAR"

def analyze_stock(symbol, start_date, end_date, investor_type):
    data = get_stock_data(symbol, start_date, end_date)
    if data.empty:
        return {
            "symbol": symbol,
            "analysis": f"No data available for {symbol} in the specified date range.",
            "recommendation": "UNCLEAR",
            "plot": None,
            "performance_metrics": {},
        }
    
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    data_summary = data.describe().to_string()
    
    indicators = get_technical_indicators(data)
    news_headlines = get_news_headlines(symbol)
    if "No recent news headlines available" in news_headlines or not news_headlines.strip():
        news_headlines = get_market_context(symbol)
    
    start_time = time.time()
    analysis = stock_analysis_chain.run(
        stock_data=data_summary,
        stock_symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        start_price=f"{start_price:.2f}",
        end_price=f"{end_price:.2f}",
        sma=f"{indicators['sma']:.2f}" if indicators['sma'] is not None else "N/A",
        rsi=f"{indicators['rsi']:.2f}" if indicators['rsi'] is not None else "N/A",
        momentum=f"{indicators['momentum']:.2f}",
        price_vs_sma=indicators['price_vs_sma'],
        news_headlines=news_headlines,
        investor_type=investor_type
    )
    end_time = time.time()
    inference_time = end_time - start_time
    recommendation = extract_recommendation(analysis)
    plot = plot_stock_data(data, symbol)
    performance_metrics = {
        "inference_time": f"{inference_time:.2f} seconds",
        "token_count": len(analysis.split()),
        "data_points": len(data)
    }
    return {
        "symbol": symbol,
        "analysis": analysis,
        "recommendation": recommendation,
        "plot": plot,
        "performance_metrics": performance_metrics,
    }

# Multi-stock support: comma-separated symbols
def gradio_interface(symbols, start_date, end_date, investor_type):
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    all_analyses = []
    all_recommendations = []
    all_plots = []
    all_inference_times = []
    all_token_counts = []
    all_data_points = []
    for symbol in symbol_list:
        result = analyze_stock(symbol, start_date, end_date, investor_type)
        all_analyses.append(f"[{symbol} - {investor_type} Investor]\n" + result["analysis"])
        all_recommendations.append(f"[{symbol}] {result['recommendation']}")
        all_plots.append(result["plot"])
        pm = result["performance_metrics"]
        all_inference_times.append(f"[{symbol}] LLM Inference Time: {pm.get('inference_time', 'N/A')}")
        all_token_counts.append(f"[{symbol}] Token Count: {pm.get('token_count', 'N/A')}")
        all_data_points.append(f"[{symbol}] Data Points: {pm.get('data_points', 'N/A')}")
    # For plots, only show the first one if multiple
    return (
        '\n\n'.join(all_analyses),
        '\n'.join(all_recommendations),
        all_plots[0] if all_plots else None,
        '\n'.join(all_inference_times),
        '\n'.join(all_token_counts),
        '\n'.join(all_data_points)
    )

# Create an enhanced Gradio interface with AMD branding
def create_interface():
    # Custom CSS for AMD branding
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* AMD Red accent color */
    .primary {
        background: linear-gradient(135deg, #ED1C24 0%, #B71C1C 100%) !important;
        border: none !important;
    }
    
    .primary:hover {
        background: linear-gradient(135deg, #B71C1C 0%, #8B0000 100%) !important;
    }
    
    /* Tab styling with AMD red */
    .tab-nav button.selected {
        color: #ED1C24 !important;
        border-bottom: 2px solid #ED1C24 !important;
    }
    
    /* Headers with AMD red accents */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }
    
    /* Input focus states with AMD red */
    input:focus, textarea:focus, select:focus {
        border-color: #ED1C24 !important;
        box-shadow: 0 0 0 2px rgba(237, 28, 36, 0.1) !important;
    }
    
    /* Links and accents */
    a {
        color: #ED1C24 !important;
    }
    
    /* Section headers */
    h3 {
        border-left: 4px solid #ED1C24 !important;
        padding-left: 12px !important;
    }
    
    .gradio-container {max-width: 1200px; margin: auto;}
    .gr-box {border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    """
    
    with gr.Blocks(title="🚀 Personalized Multi-Stock AI Analysis Tool with Investor Profiles", theme=gr.themes.Soft(), css=custom_css) as interface:
        # Header with AMD logo in top right corner
        gr.HTML("""
            <div style="position: relative; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-bottom: 20px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" alt="AMD Logo" style="position: absolute; top: 15px; right: 20px; height: 35px; width: auto;" />
                <div style="padding-right: 120px;">
                    <h1 style="margin: 0; color: #2c3e50; font-size: 2.2em; font-weight: 700;">🚀 Personalized Multi-Stock AI Analysis Tool</h1>
                    <h3 style="margin: 5px 0 0 0; color: #ED1C24; font-size: 1.2em; font-weight: 600;">Powered by AMD MI300X with Investor Profile Customization</h3>
                </div>
            </div>
        """)
        
        gr.Markdown("""
            **AI-Powered Stock Analysis with Personalized Investment Profiles**
            
            Get comprehensive stock analysis tailored to your investment style and risk tolerance. 
            This tool provides personalized recommendations based on your investor profile, powered by 
            AMD's advanced GPU acceleration.

            ### Key Features:
            - **Multi-Stock Analysis**: Analyze multiple stocks simultaneously
            - **Personalized Profiles**: Tailored analysis for Conservative, Moderate, Aggressive, and Day Trader styles
            - **Technical Indicators**: SMA, RSI, momentum analysis
            - **Real-time News Integration**: Latest market headlines and sentiment
            - **GPU Acceleration**: Fast processing with AMD hardware
            
            Enter one or more stock symbols, date range, and select your investor profile to get personalized recommendations.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Configuration")
                
                symbols_input = gr.Textbox(
                    label="Stock Symbol(s) (e.g., AAPL, MSFT)", 
                    placeholder="Enter one or more stock symbols, separated by commas...",
                    lines=2
                )
                
                start_date_input = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)", 
                    placeholder="Enter start date..."
                )
                
                end_date_input = gr.Textbox(
                    label="End Date (YYYY-MM-DD)", 
                    placeholder="Enter end date..."
                )
                
                investor_type_input = gr.Dropdown(
                    choices=["Conservative", "Moderate", "Aggressive", "Day Trader"],
                    label="Investor Type",
                    value="Moderate",
                    info="Select your investment profile for personalized recommendations"
                )
                
                analyze_btn = gr.Button("📈 Analyze Stocks", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analysis Results")
                
                with gr.Tabs():
                    with gr.TabItem("🤖 AI Analysis"):
                        ai_analysis_output = gr.Textbox(
                            label="AI Analysis", 
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.TabItem("💡 Recommendations"):
                        recommendations_output = gr.Textbox(
                            label="Investment Recommendations",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.TabItem("📊 Price Chart"):
                        chart_output = gr.Plot(label="Stock Price Chart (First Symbol)")
                    
                    with gr.TabItem("⚡ Performance Metrics"):
                        with gr.Row():
                            inference_time_output = gr.Textbox(label="LLM Inference Time(s)", interactive=False)
                            token_count_output = gr.Textbox(label="Token Count(s)", interactive=False)
                            data_points_output = gr.Textbox(label="Data Points Analyzed", interactive=False)
        
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
        
        # Example section
        gr.Markdown("""
            ### 💡 Example Usage
            1. Enter stock symbols (e.g., "AAPL, MSFT, GOOGL")
            2. Set your analysis date range
            3. Select your investor profile based on your risk tolerance
            4. Click "Analyze Stocks" to get personalized analysis
            5. Review AI-generated insights tailored to your investment style
            
            ### 📊 Investor Profiles
            - **Conservative**: Capital preservation, dividend focus, low risk
            - **Moderate**: Balanced growth and stability approach
            - **Aggressive**: High growth potential, higher risk tolerance
            - **Day Trader**: Short-term technical analysis focus
        """)
    
    return interface

iface = create_interface()

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")