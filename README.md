# Quantitative Trading Analysis Dashboard

A comprehensive Python-based quantitative trading platform featuring backtesting, risk management, and interactive visualization dashboard for financial market analysis.

## Features

- **Data Collection**: Real-time and historical market data acquisition using yFinance
- **Trading Strategies**: Implementation of momentum and mean reversion strategies
- **Backtesting Engine**: Comprehensive backtesting framework with performance metrics
- **Risk Management**: Portfolio risk assessment and Value-at-Risk (VaR) calculations
- **Interactive Dashboard**: Streamlit-based web application for strategy analysis
- **Jupyter Notebooks**: Data exploration and strategy development workflows

## Project Structure

```
quantdashboard/
├── trading.quant mvp/
│   ├── data/                    # Data storage
│   │   ├── market_data/        # Raw market data
│   │   └── processed/          # Processed datasets
│   ├── src/
│   │   ├── backtesting/        # Backtesting engine and metrics
│   │   ├── data_collection/    # Data acquisition modules
│   │   ├── models/             # Trading strategy implementations
│   │   ├── notebook/           # Jupyter notebooks for analysis
│   │   ├── risk_management/    # Risk assessment tools
│   │   ├── tests/              # Unit tests
│   │   └── visualization/      # Dashboard and plotting utilities
│   ├── requirements.txt        # Python dependencies
│   └── venv/                   # Virtual environment
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pnehete23/Quant-Trade-Analysis-Dashboard.git
cd quantdashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r "trading.quant mvp/requirements.txt"
```

## Usage

### Running the Dashboard
```bash
streamlit run "trading.quant mvp/src/visualization/dashboard.py"
```

### Jupyter Notebooks
Navigate to the notebook directory and start Jupyter:
```bash
cd "trading.quant mvp/src/notebook"
jupyter notebook
```

### Key Modules

#### Data Collection
- `yfinance_collector.py`: Real-time market data acquisition
- `data_processor.py`: Data cleaning and preprocessing

#### Trading Strategies
- `momentum_strategy.py`: Momentum-based trading strategy
- `mean_reversion.py`: Mean reversion strategy implementation

#### Backtesting
- `backtest_engine.py`: Core backtesting framework
- `performance_metrics.py`: Strategy performance evaluation

#### Risk Management
- `portfolio_risk.py`: Portfolio risk assessment
- `var_calculator.py`: Value-at-Risk calculations

## Key Dependencies

- **Data Analysis**: pandas, numpy, scipy, scikit-learn
- **Financial Data**: yfinance, quantlib, zipline-reloaded
- **Visualization**: plotly, matplotlib, seaborn, streamlit
- **Development**: pytest, jupyter, black, flake8

## Getting Started

1. **Data Exploration**: Start with `01_data_exploration.ipynb` to understand market data
2. **Strategy Development**: Use `02_strategy_development.ipynb` to build and test strategies
3. **Backtesting Analysis**: Run `03_backtesting_analysis.ipynb` for performance evaluation
4. **Risk Assessment**: Analyze portfolio risk with `04_risk_assessment.ipynb`
5. **Dashboard**: Launch the interactive dashboard for real-time analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-strategy`)
3. Commit your changes (`git commit -am 'Add new trading strategy'`)
4. Push to the branch (`git push origin feature/new-strategy`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading financial instruments carries risk and may result in financial loss. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## Contact

For questions or support, please open an issue on GitHub or contact the project maintainer.