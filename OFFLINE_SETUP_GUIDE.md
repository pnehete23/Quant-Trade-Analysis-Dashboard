# Offline Setup Guide - Quantitative Trading Dashboard

## Prerequisites for Offline Installation

### System Requirements
- Windows 10/11
- Python 3.8+ (3.11 recommended)
- At least 4GB RAM
- 2GB free disk space

### Required Downloads (while online)

1. **Python Installer**: Download Python 3.11 from python.org
2. **pip packages**: Use the included `current_env_requirements.txt`
3. **Git** (optional): For version control

## Offline Installation Steps

### 1. Python Environment Setup
```bash
# Create virtual environment
python -m venv quantdash_env

# Activate environment (Windows)
quantdash_env\Scripts\activate

# For macOS/Linux
source quantdash_env/bin/activate
```

### 2. Install Dependencies from Offline Packages

**Option A: Using pip freeze file (recommended)**
```bash
pip install -r current_env_requirements.txt --no-index --find-links ./offline_packages
```

**Option B: Using requirements.txt**
```bash
pip install -r requirements.txt --no-index --find-links ./offline_packages
```

### 3. Verify Installation
```bash
python -c "import pandas, numpy, yfinance, streamlit; print('All packages imported successfully')"
```

## Directory Structure After Setup
```
quantdashboard/
├── trading.quant mvp/
│   ├── src/                     # Source code
│   ├── data/                    # Data storage
│   ├── requirements.txt         # Original requirements
│   ├── current_env_requirements.txt  # Exact frozen requirements
│   └── venv/                    # Virtual environment
├── quantdash_env/              # Offline environment
├── offline_packages/           # Downloaded packages (if using)
└── OFFLINE_SETUP_GUIDE.md     # This guide
```

## Running the Application Offline

### 1. Start the Dashboard
```bash
# Activate environment
quantdash_env\Scripts\activate

# Navigate to project
cd "trading.quant mvp"

# Run dashboard
streamlit run src/visualization/dashboard.py
```

### 2. Access Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Navigate to: src/notebook/
# Open any .ipynb file
```

## Key Components Available Offline

### Data Collection
- **Historical Data**: yfinance for market data
- **Data Processing**: pandas, numpy for analysis

### Trading Strategies
- **Momentum Strategy**: `src/models/momentum_strategy.py`
- **Mean Reversion**: `src/models/mean_reversion.py`

### Backtesting
- **Engine**: `src/backtesting/backtest_engine.py`
- **Metrics**: `src/backtesting/performance_metrics.py`

### Risk Management
- **Portfolio Risk**: `src/risk_management/portfolio_risk.py`
- **VaR Calculator**: `src/risk_management/var_calculator.py`

### Visualization
- **Dashboard**: `src/visualization/dashboard.py`
- **Plots**: `src/visualization/plots.py`

## Troubleshooting Offline Issues

### 1. Missing Packages
- Check `current_env_requirements.txt` for exact versions
- Ensure all dependencies are in offline package directory

### 2. Data Access
- Application can work with previously downloaded data
- Store sample datasets in `data/market_data/` for testing

### 3. Performance Issues
- Reduce data size for faster processing offline
- Use cached results when possible

## Sample Data for Offline Testing

Create sample files in `data/market_data/`:
- `AAPL_daily.csv`
- `SPY_daily.csv`
- `portfolio_data.csv`

## Notes
- Internet connection required only for live market data
- All analysis and backtesting can run completely offline
- Dashboard functionality preserved without live data feeds
- Jupyter notebooks work fully offline with saved data

## Contact
For offline setup issues, check the main README.md or project documentation.