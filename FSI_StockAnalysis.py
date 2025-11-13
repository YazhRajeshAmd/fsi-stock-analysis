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
llm = Ollama(model="llama3.1:8b", temperature=0.3)

# Updated prompt template focused on comprehensive AI analysis
stock_analysis_prompt = PromptTemplate(
    input_variables=["stock_data", "stock_symbol", "start_date", "end_date", "start_price", "end_price", 
                    "sma", "rsi", "news_headlines", "momentum", "price_vs_sma", "investor_type"],
    template="""
You are an advanced AI financial analyst powered by AMD MI300X GPU and ROCm platform, providing comprehensive stock analysis with cutting-edge computational capabilities.

Stock Analysis Data:
Stock Symbol: {stock_symbol}
Date Range: {start_date} to {end_date}
Starting Price: ${start_price}
Ending Price: ${end_price}
Investor Type: {investor_type}

Stock Data Summary: {stock_data}

Technical Indicators:
SMA (20): {sma}
RSI (14): {rsi}
Price Momentum: {momentum}%
Price vs SMA: {price_vs_sma}

Recent News Headlines: {news_headlines}

COMPREHENSIVE AI ANALYSIS FRAMEWORK:

For {investor_type} Investor Profile:

Conservative Investor:
- Prioritizes capital preservation and steady income
- Prefers established companies with strong fundamentals
- Low risk tolerance, seeks dividend-paying stocks
- Focus on stability metrics and defensive sectors

Moderate Investor:
- Balanced approach between growth and stability
- Willing to accept moderate risk for better returns
- Diversified portfolio strategy
- Growth potential with reasonable risk assessment

Aggressive Investor:
- High risk tolerance, seeks maximum capital appreciation
- Comfortable with volatility and market fluctuations
- Focus on growth stocks and emerging opportunities
- Innovation-driven investment decisions

Day Trader:
- Short-term trading focus (minutes to days)
- Technical analysis and momentum-driven decisions
- High-frequency trading considerations
- Volume and volatility analysis

PROVIDE COMPREHENSIVE ANALYSIS INCLUDING:

1. **Technical Analysis Deep Dive:**
   - Price trend analysis and pattern recognition
   - Moving averages and momentum indicators
   - Support and resistance levels
   - Volume analysis and market sentiment

2. **Fundamental Analysis:**
   - Company financial health assessment
   - Industry position and competitive landscape
   - Revenue growth trends and profitability metrics
   - Market capitalization and valuation ratios

3. **Market Sentiment & News Impact:**
   - Recent news sentiment analysis
   - Market conditions affecting the stock
   - Sector performance comparison
   - Economic indicators influence

4. **Risk Assessment:**
   - Volatility analysis and risk metrics
   - Market correlation and beta analysis
   - Downside protection and stop-loss levels
   - Portfolio diversification considerations

5. **Price Targets & Projections:**
   - Technical price targets based on chart patterns
   - Analyst consensus and price predictions
   - Scenario analysis (bull/bear/base cases)
   - Time horizon considerations for {investor_type}

6. **Investment Strategy Recommendations:**
   - Position sizing recommendations
   - Entry and exit strategies
   - Risk management protocols
   - Portfolio allocation suggestions

Provide detailed analysis with specific data points, percentages, and actionable insights tailored for {investor_type} investment style.

End with a clear, confident recommendation:
"AI RECOMMENDATION FOR {investor_type}: [BUY/SELL/HOLD]"

Include your confidence level (High/Medium/Low) and key reasoning behind the recommendation.

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
    # Try multiple patterns to catch the recommendation, focusing on AI recommendations
    patterns = [
        r"AI RECOMMENDATION FOR [^:]+:\s*(BUY|SELL|HOLD)",
        r"RECOMMENDATION FOR [^:]+:\s*(BUY|SELL|HOLD)",
        r"RECOMMENDATION:\s*(BUY|SELL|HOLD)",
        r"(BUY|SELL|HOLD)\s*(?:recommendation|decision|action)",
        r"My recommendation.*?is\s*(BUY|SELL|HOLD)",
        r"I recommend.*?(BUY|SELL|HOLD)",
        r"Final.*?recommendation.*?(BUY|SELL|HOLD)",
        r"GRAHAM-INSPIRED RECOMMENDATION FOR [^:]+:\s*(BUY|SELL|HOLD)"
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
    print(f"\n🕒 User-selected date range:")
    print(f"   Start Date: {start_date}")
    print(f"   End Date:   {end_date}")
    print(f"   Investor Type: {investor_type}")
    print(f"   Symbols: {symbols}\n")
    # For plots, only show the first one if multiple
    return (
        '\n\n'.join(all_analyses),
        '\n'.join(all_recommendations),
        all_plots[0] if all_plots else None,
        '\n'.join(all_inference_times),
        '\n'.join(all_token_counts),
        '\n'.join(all_data_points)
    )

# Create an enhanced Gradio interface
def create_interface():
    # Custom CSS for Teal branding
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Arial:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Arial', Arial, sans-serif !important;
    }
    
    /* Teal accent color - PMS 3115 C */
    .primary {
        background: linear-gradient(135deg, #00C2DE 0%, #008AA8 100%) !important;
        border: none !important;
    }
    
    .primary:hover {
        background: linear-gradient(135deg, #008AA8 0%, #006A80 100%) !important;
    }
    
    /* Tab styling with Teal */
    .tab-nav button.selected {
        color: #00C2DE !important;
        border-bottom: 2px solid #00C2DE !important;
    }
    
    /* Headers with Teal accents */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }
    
    /* Input focus states with Teal */
    input:focus, textarea:focus, select:focus {
        border-color: #00C2DE !important;
        box-shadow: 0 0 0 2px rgba(0, 194, 222, 0.1) !important;
    }
    
    /* Links and accents */
    a {
        color: #00C2DE !important;
    }
    
    /* Section headers */
    h3 {
        border-left: 4px solid #00C2DE !important;
        padding-left: 12px !important;
    }
    
    /* Increased container width to accommodate horizontal tabs */
    .gradio-container {
        max-width: 1600px !important; 
        margin: auto !important;
        width: 100% !important;
    }
    
    /* Simplified tab styling - ensure visibility */
    .gradio-tabs {
        width: 100% !important;
        background: transparent !important;
    }
    
    /* Tab navigation styling */
    .gradio-tabs .tab-nav,
    .gradio-tabs > div:first-child {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 8px !important;
        background: #f8f9fa !important;
        padding: 10px !important;
        border-radius: 10px !important;
        margin-bottom: 15px !important;
        width: 100% !important;
    }
    
    /* Individual tab buttons */
    .gradio-tabs .tab-nav button,
    .gradio-tabs > div:first-child > button {
        flex: 1 !important;
        min-width: 160px !important;
        max-width: 220px !important;
        padding: 10px 12px !important;
        border-radius: 6px !important;
        border: 2px solid #e0e0e0 !important;
        background: white !important;
        color: #666 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: block !important;
        visibility: visible !important;
    }
    
    .gradio-tabs .tab-nav button:hover {
        border-color: #00C2DE !important;
        color: #00C2DE !important;
        background: rgba(0, 194, 222, 0.05) !important;
    }
    
    .gradio-tabs .tab-nav button.selected {
        background: #00C2DE !important;
        color: white !important;
        border-color: #00C2DE !important;
    }
    
    /* Hide dropdown menu and force horizontal display */
    .gradio-tabs .tab-nav .tab-nav-button,
    .gradio-tabs button[aria-label="More tabs"],
    .gradio-tabs .tab-nav button:last-child[style*="display: none"] {
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
    
    .gr-box {border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    """
    
    with gr.Blocks(title="AMD Instinct MI3xx ROCm-Powered Financial Analysis Demo", theme=gr.themes.Soft(), css=custom_css) as interface:
        # Header with AMD logo in top right corner
        gr.HTML("""
            <div style="position: relative; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-bottom: 20px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" alt="AMD Logo" style="position: absolute; top: 15px; right: 20px; height: 35px; width: auto;" />
                <div style="padding-right: 120px;">
                    <h1 style="margin: 0; color: #2c3e50; font-size: 2.2em; font-weight: 700; font-family: Arial, sans-serif;"> AMD Instinct ROCm-Powered Financial Analysis Demo</h1>
                    <h3 style="margin: 5px 0 0 0; color: #00C2DE; font-size: 1.2em; font-weight: 600; font-family: Arial, sans-serif;">Powered by ROCm Platform </h3>
                </div>
            </div>
        """)
        
        gr.Markdown("""
            **Advanced AI-Driven Stock Analysis on AMD Hardware**
            
            This cutting-edge financial analysis tool leverages AMD's Instinct GPU architecture with ROCm platform 
            to deliver high-performance AI-driven stock analysis with modern GPU acceleration.

            ### Key Features:
            - **AMD Instinct Architecture**: High HBM3 memory for complex financial modeling
            - **ROCm Software Stack**: Open-source GPU acceleration platform
            - **Real-time Processing**: GPU-accelerated technical indicators and market data
            - **Multi-Stock Portfolio Analysis**: Parallel processing capabilities

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
                            value="2025-08-13",
                            placeholder="2025-01-01",
                            info="Enter end date in YYYY-MM-DD format"
                        )
                
                investor_type_input = gr.Dropdown(
                    choices=["Conservative", "Moderate", "Aggressive", "Day Trader"],
                    label="Investor Type",
                    value="Moderate",
                    info="Select your investment profile for personalized recommendations"
                )
                
                analyze_btn = gr.Button("🚀 Analyze Stocks", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analysis Results")
                
                with gr.Tabs():
                    with gr.TabItem("🤖 AI Technical Analysis"):
                        ai_analysis_output = gr.Textbox(
                            label="AI Technical & Market Analysis", 
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
        
        # Performance Metrics moved down below main interface
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚡ System Performance Metrics")
                with gr.Row():
                    inference_time_output = gr.Textbox(label="LLM Inference Time (s)", interactive=False, scale=1)
                    token_count_output = gr.Textbox(label="Token Count", interactive=False, scale=1)
                    data_points_output = gr.Textbox(label="Data Points Analyzed", interactive=False, scale=1)
        
        # Add JavaScript for horizontal tabs
        gr.HTML("""
        <script>
        function ensureTabsVisible() {
            // Find all tab containers
            const tabContainers = document.querySelectorAll('.gradio-tabs');
            tabContainers.forEach(container => {
                // Find the tab navigation area
                const tabNav = container.querySelector('div:first-child');
                if (tabNav) {
                    // Simple flex layout
                    tabNav.style.display = 'flex';
                    tabNav.style.flexWrap = 'nowrap';
                    tabNav.style.gap = '8px';
                    tabNav.style.width = '100%';
                    
                    // Make all buttons visible
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
        
        // Run after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(ensureTabsVisible, 500);
            setTimeout(ensureTabsVisible, 1500);
        });
        </script>
        """)
        
        # Add additional CSS to ensure tabs display properly
        gr.HTML("""
        <style>
        /* Force tab navigation to be visible */
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
        
        /* Make tab buttons clearly visible */
        .gradio-tabs .tab-nav button,
        .gradio-tabs > div:first-child > button {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            flex: 1 !important;
            min-width: 150px !important;
            max-width: 200px !important;
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
        
        /* Tab button hover and active states */
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
        
        /* Hide dropdown buttons completely */
        .gradio-tabs button[aria-label*="More"],
        .gradio-tabs .tab-nav-button {
            display: none !important;
        }
        
        /* Ensure tab content area has proper styling */
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
        
        # Example section
        gr.Markdown("""
            ### 💡 Example Usage
            1. Enter stock symbols (e.g., "AAPL, MSFT, GOOGL")
            2. Set your analysis date range
            3. Select your investor profile (Conservative, Moderate, Aggressive, Day Trader)
            4. Click "Analyze Stocks" to start GPU-accelerated analysis
            5. Review results across different analysis perspectives

            ### Disclaimer
            This tool is for educational purposes only. Always conduct your own research and consult with a financial advisor before making investment decisions.
        """)
    
    return interface

iface = create_interface()

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
