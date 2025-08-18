# src/data_collection/yfinance_collector.py
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class YFinanceCollector:
    """
    Professional Yahoo Finance data collector with advanced features
    
    Features:
    - Bulk data collection with rate limiting
    - Automatic retry mechanism for failed requests
    - Data validation and error handling
    - Support for multiple asset classes
    - Historical and real-time data collection
    - Fundamental data integration
    """
    
    def __init__(self, max_workers: int = 5, request_delay: float = 0.1):
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.session = requests.Session()
        
    def get_stock_data(self, ticker: str, start_date: str, end_date: str, 
                      interval: str = '1d', auto_adjust: bool = True, 
                      prepost: bool = False) -> pd.DataFrame:
        """
        Download stock data for a single ticker
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            auto_adjust: Automatically adjust for splits and dividends
            prepost: Include pre and post market data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                repair=True
            )
            
            if data.empty:
                print(f"Warning: No data found for {ticker}")
                return pd.DataFrame()
            
            # Add ticker column for identification
            data['Ticker'] = ticker
            
            # Calculate additional metrics
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close']).diff()
            data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            
            time.sleep(self.request_delay)  # Rate limiting
            return data
            
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, tickers: List[str], start_date: str, end_date: str,
                           interval: str = '1d', max_retries: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers concurrently
        
        Args:
            tickers: List of stock symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            max_retries: Maximum retry attempts for failed requests
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        failed_tickers = []
        
        def download_ticker(ticker):
            for attempt in range(max_retries):
                try:
                    data = self.get_stock_data(ticker, start_date, end_date, interval)
                    if not data.empty:
                        return ticker, data
                    else:
                        print(f"Attempt {attempt + 1}: No data for {ticker}")
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
            return ticker, None
        
        # Use ThreadPoolExecutor for concurrent downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {executor.submit(download_ticker, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker, data = future.result()
                if data is not None:
                    results[ticker] = data
                    print(f"Successfully downloaded data for {ticker}")
                else:
                    failed_tickers.append(ticker)
                    print(f"Failed to download data for {ticker}")
        
        if failed_tickers:
            print(f"Failed to download data for: {failed_tickers}")
        
        return results
    
    def get_fundamental_data(self, ticker: str) -> Dict:
        """
        Get fundamental data for a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with fundamental data
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Get various fundamental data
            info = ticker_obj.info
            financials = ticker_obj.financials
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
            earnings = ticker_obj.earnings
            
            # Extract key metrics
            key_metrics = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_value': info.get('enterpriseValue'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'return_on_assets': info.get('returnOnAssets'),
                'return_on_equity': info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'current_ratio': info.get('currentRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'insider_percent': info.get('heldPercentInsiders'),
                'institution_percent': info.get('heldPercentInstitutions')
            }
            
            return {
                'ticker': ticker,
                'basic_info': info,
                'key_metrics': key_metrics,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'earnings': earnings,
                'collection_date': datetime.now()
            }
            
        except Exception as e:
            print(f"Error getting fundamental data for {ticker}: {str(e)}")
            return {}
    
    def get_options_data(self, ticker: str, expiration_date: str = None) -> Dict:
        """
        Get options data for a ticker
        
        Args:
            ticker: Stock symbol
            expiration_date: Specific expiration date (YYYY-MM-DD)
            
        Returns:
            Dictionary with options data
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Get available expiration dates
            expiration_dates = ticker_obj.options
            
            if not expiration_dates:
                print(f"No options data available for {ticker}")
                return {}
            
            if expiration_date is None:
                expiration_date = expiration_dates[0]  # Use nearest expiration
            elif expiration_date not in expiration_dates:
                print(f"Expiration date {expiration_date} not available for {ticker}")
                expiration_date = expiration_dates[0]
            
            # Get options chain
            options_chain = ticker_obj.option_chain(expiration_date)
            calls = options_chain.calls
            puts = options_chain.puts
            
            return {
                'ticker': ticker,
                'expiration_date': expiration_date,
                'available_expirations': list(expiration_dates),
                'calls': calls,
                'puts': puts,
                'collection_date': datetime.now()
            }
            
        except Exception as e:
            print(f"Error getting options data for {ticker}: {str(e)}")
            return {}
    
    def get_dividend_data(self, ticker: str, start_date: str = None) -> pd.DataFrame:
        """
        Get dividend history for a ticker
        
        Args:
            ticker: Stock symbol
            start_date: Start date for dividend history
            
        Returns:
            DataFrame with dividend data
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            
            if start_date:
                dividends = ticker_obj.dividends[start_date:]
            else:
                dividends = ticker_obj.dividends
            
            if dividends.empty:
                print(f"No dividend data found for {ticker}")
                return pd.DataFrame()
            
            # Calculate dividend metrics
            dividend_df = dividends.to_frame('Dividend')
            dividend_df['Ticker'] = ticker
            dividend_df['Cumulative_Dividend'] = dividend_df['Dividend'].cumsum()
            
            # Annual dividend yield calculation
            annual_dividends = dividend_df.groupby(dividend_df.index.year)['Dividend'].sum()
            dividend_df['Annual_Dividend'] = dividend_df.index.map(
                lambda x: annual_dividends.get(x.year, 0)
            )
            
            return dividend_df
            
        except Exception as e:
            print(f"Error getting dividend data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_sector_industry_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get sector and industry information for multiple tickers
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            DataFrame with sector/industry data
        """
        sector_data = []
        
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                sector_data.append({
                    'Ticker': ticker,
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', 'Unknown'),
                    'Country': info.get('country', 'Unknown'),
                    'Market_Cap': info.get('marketCap', 0),
                    'Employee_Count': info.get('fullTimeEmployees', 0),
                    'Business_Summary': info.get('longBusinessSummary', '')
                })
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                print(f"Error getting sector data for {ticker}: {str(e)}")
                sector_data.append({
                    'Ticker': ticker,
                    'Sector': 'Error',
                    'Industry': 'Error',
                    'Country': 'Error',
                    'Market_Cap': 0,
                    'Employee_Count': 0,
                    'Business_Summary': ''
                })
        
        return pd.DataFrame(sector_data)
    
    def get_earnings_calendar(self, ticker: str) -> pd.DataFrame:
        """
        Get earnings calendar data
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with earnings data
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            calendar = ticker_obj.calendar
            
            if calendar is None or calendar.empty:
                print(f"No earnings calendar data for {ticker}")
                return pd.DataFrame()
            
            calendar['Ticker'] = ticker
            return calendar
            
        except Exception as e:
            print(f"Error getting earnings calendar for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def create_market_snapshot(self, tickers: List[str]) -> Dict:
        """
        Create a comprehensive market snapshot
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary with market snapshot data
        """
        snapshot_date = datetime.now()
        
        # Get current price data
        price_data = {}
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period="5d")  # Last 5 days
                info = ticker_obj.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    price_data[ticker] = {
                        'current_price': current_price,
                        'previous_close': prev_close,
                        'change': change,
                        'change_percent': change_pct,
                        'volume': hist['Volume'].iloc[-1],
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'sector': info.get('sector', 'Unknown')
                    }
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                print(f"Error in snapshot for {ticker}: {str(e)}")
        
        # Create summary statistics
        if price_data:
            changes = [data['change_percent'] for data in price_data.values()]
            summary_stats = {
                'total_stocks': len(price_data),
                'gainers': len([x for x in changes if x > 0]),
                'losers': len([x for x in changes if x < 0]),
                'unchanged': len([x for x in changes if x == 0]),
                'avg_change': np.mean(changes),
                'max_gainer': max(changes) if changes else 0,
                'max_loser': min(changes) if changes else 0
            }
        else:
            summary_stats = {}
        
        return {
            'snapshot_date': snapshot_date,
            'price_data': price_data,
            'summary_stats': summary_stats,
            'market_indices': self._get_market_indices()
        }
    
    def _get_market_indices(self) -> Dict:
        """Get major market indices data"""
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX'
        }
        
        index_data = {}
        for symbol, name in indices.items():
            try:
                ticker_obj = yf.Ticker(symbol)
                hist = ticker_obj.history(period="2d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - previous
                    change_pct = (change / previous) * 100 if previous != 0 else 0
                    
                    index_data[symbol] = {
                        'name': name,
                        'current': current,
                        'change': change,
                        'change_percent': change_pct
                    }
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                print(f"Error getting index data for {symbol}: {str(e)}")
        
        return index_data
    
    def validate_tickers(self, tickers: List[str]) -> Dict[str, bool]:
        """
        Validate if tickers exist and have data
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary mapping ticker to validation status
        """
        validation_results = {}
        
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                # Try to get recent data
                hist = ticker_obj.history(period="5d")
                info = ticker_obj.info
                
                # Check if we got valid data
                is_valid = (
                    not hist.empty and 
                    'regularMarketPrice' in info and 
                    info.get('regularMarketPrice') is not None
                )
                
                validation_results[ticker] = is_valid
                
            except Exception as e:
                print(f"Validation failed for {ticker}: {str(e)}")
                validation_results[ticker] = False
        
        return validation_results

# Example usage and testing
if __name__ == "__main__":
    # Initialize collector
    collector = YFinanceCollector(max_workers=3, request_delay=0.2)
    
    # Test single stock download
    print("=== Testing Single Stock Download ===")
    aapl_data = collector.get_stock_data("AAPL", "2023-01-01", "2024-01-01")
    print(f"AAPL data shape: {aapl_data.shape}")
    print(f"AAPL data columns: {list(aapl_data.columns)}")
    print(f"AAPL recent data:\n{aapl_data.tail(3)}")
    
    # Test multiple stocks download
    print("\n=== Testing Multiple Stocks Download ===")
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    multi_data = collector.get_multiple_stocks(tickers, "2023-01-01", "2024-01-01")
    print(f"Downloaded data for {len(multi_data)} tickers")
    for ticker, data in multi_data.items():
        print(f"{ticker}: {data.shape[0]} rows, Latest close: ${data['Close'].iloc[-1]:.2f}")
    
    # Test fundamental data
    print("\n=== Testing Fundamental Data ===")
    aapl_fundamentals = collector.get_fundamental_data("AAPL")
    if aapl_fundamentals:
        key_metrics = aapl_fundamentals['key_metrics']
        print("AAPL Key Metrics:")
        for metric, value in key_metrics.items():
            if value is not None and metric in ['market_cap', 'pe_ratio', 'beta', 'dividend_yield']:
                print(f"  {metric}: {value}")
    
    # Test dividend data
    print("\n=== Testing Dividend Data ===")
    aapl_dividends = collector.get_dividend_data("AAPL", "2023-01-01")
    if not aapl_dividends.empty:
        print(f"AAPL Dividends shape: {aapl_dividends.shape}")
        print(f"Recent dividends:\n{aapl_dividends.tail(3)}")
    
    # Test sector/industry data
    print("\n=== Testing Sector/Industry Data ===")
    sector_data = collector.get_sector_industry_data(["AAPL", "GOOGL", "JPM"])
    print(f"Sector data:\n{sector_data[['Ticker', 'Sector', 'Industry', 'Market_Cap']]}")
    
    # Test market snapshot
    print("\n=== Testing Market Snapshot ===")
    snapshot = collector.create_market_snapshot(["AAPL", "GOOGL", "MSFT"])
    print(f"Snapshot date: {snapshot['snapshot_date']}")
    print(f"Summary stats: {snapshot['summary_stats']}")
    
    # Test ticker validation
    print("\n=== Testing Ticker Validation ===")
    test_tickers = ["AAPL", "GOOGL", "INVALID123", "MSFT"]
    validation = collector.validate_tickers(test_tickers)
    print("Validation results:")
    for ticker, is_valid in validation.items():
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"  {ticker}: {status}")
    
    print("\n=== Data Collection Complete ===")