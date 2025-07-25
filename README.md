# Financial Stock Intelligence (FSI)

🚀 **AI-Powered Stock Analysis with Technical Indicators and LLM Insights**

A sophisticated financial analysis tool that combines real-time stock data, technical indicators, and Large Language Model (LLM) analysis to provide comprehensive stock insights.

## 🎯 Features

- **Real-time Stock Data**: Fetches live stock prices using Yahoo Finance API
- **Technical Analysis**: 
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Momentum calculations
  - Price vs SMA comparisons
- **AI-Powered Analysis**: Uses Llama 3.1 70B model via Ollama for intelligent stock insights
- **Interactive Web Interface**: Beautiful Gradio-based UI for easy interaction
- **Historical Data Visualization**: Charts and graphs for trend analysis
- **News Integration**: Incorporates relevant financial news for context

## 🛠️ Technology Stack

- **Python 3.8+**
- **LangChain**: LLM orchestration and prompt management
- **Ollama**: Local LLM inference (Llama 3.1 70B)
- **yfinance**: Real-time stock data
- **Gradio**: Web interface
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** and pull the Llama 3.1 70B model:
   ```bash
   ollama pull llama3.1:70b
   ```

2. **Python 3.8+** installed on your system

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fsi-stock-analysis.git
   cd fsi-stock-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama server:
   ```bash
   ollama serve
   ```

4. Run the application:
   ```bash
   python fsi_code.py
   ```

5. Open your browser and navigate to the provided Gradio URL (typically `http://127.0.0.1:7860`)

## 📊 Usage

1. **Enter Stock Symbol**: Input any valid stock ticker (e.g., AAPL, GOOGL, TSLA)
2. **Set Date Range**: Choose your analysis period
3. **Get AI Analysis**: Click "Analyze Stock" for comprehensive insights
4. **Review Results**: 
   - Technical indicators and charts
   - AI-generated analysis and recommendations
   - Risk assessment and market context

## 🎯 Example Analysis

The system provides detailed analysis including:
- **Price Movement**: Trend analysis and momentum
- **Technical Indicators**: RSI, SMA, and support/resistance levels
- **Market Context**: News sentiment and market conditions
- **AI Insights**: LLM-generated recommendations and risk assessment

## 🔧 Configuration

- **LLM Model**: Default uses `llama3.1:70b` (configurable in `fsi_code.py`)
- **Temperature**: Set to 0.3 for balanced creativity/accuracy
- **Technical Indicators**: Customizable periods and parameters

## 📁 Project Structure

```
fsi/
├── fsi_code.py          # Main application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── venv/              # Virtual environment (optional)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Not financial advice. Always consult with qualified financial advisors before making investment decisions.

## 🔗 Related Projects

- [MRI Analysis Tool](https://github.com/yourusername/mri-analysis-tool)
- [ROCm RAG Assistant](https://github.com/yourusername/rocm-rag-assistant)

---

**Built with ❤️ using AMD ROCm and open-source AI**
