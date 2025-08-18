# src/data_collection/data_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
from datetime import datetime
from sqlalchemy import create_engine
import os
from scipy import stats
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data processing pipeline for financial market data
    
    Features:
    - Data cleaning and validation
    - Missing data handling with multiple methods
    - Outlier detection and treatment
    - Technical indicator calculation
    - Data normalization and scaling
    - Database integration for persistence
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.engine = None
        self._setup_database()
    
    def _setup_database(self):
        """Initialize SQLite database for data storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine = create_engine(f'sqlite:///{self.db_path}')
    
    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate price data
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            # Fill missing price columns with Close if available
            if 'Close' in df.columns:
                for col in ['Open', 'High', 'Low']:
                    if col not in df.columns:
                        df[col] = df['Close']
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
        
        # Remove rows where all price data is missing
        price_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
        df = df.dropna(subset=price_cols, how='all')
        
        if len(df) == 0:
            raise ValueError("No valid price data found after cleaning")
        
        # Data validation rules
        df = self._validate_ohlc_data(df)
        df = self._handle_zero_volume(df)
        df = self._detect_and_fix_splits(df)
        
        return df
    
    def _validate_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships and fix inconsistencies"""
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Fix High values
            df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
            
            # Fix Low values  
            df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
            
            # Remove rows with negative prices
            price_cols = ['Open', 'High', 'Low', 'Close']
            df = df[(df[price_cols] > 0).all(axis=1)]
        
        return df
    
    def _handle_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle zero or missing volume data"""
        if 'Volume' in df.columns:
            # Replace zero volume with small positive number
            df.loc[df['Volume'] <= 0, 'Volume'] = 1
            
            # Fill missing volume with median of surrounding periods
            df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(10, center=True).median())
            df['Volume'] = df['Volume'].fillna(df['Volume'].median())
        
        return df
    
    def _detect_and_fix_splits(self, df: pd.DataFrame, split_threshold: float = 0.5) -> pd.DataFrame:
        """Detect and adjust for stock splits"""
        if 'Close' not in df.columns:
            return df
        
        # Calculate daily returns
        returns = df['Close'].pct_change()
        
        # Detect potential splits (large negative returns)
        potential_splits = returns < -split_threshold
        
        if potential_splits.any():
            print(f"Detected {potential_splits.sum()} potential stock splits")
            
            # For this implementation, we'll flag them but not auto-adjust
            # In production, you'd want more sophisticated split detection
            df['potential_split'] = potential_splits
        
        return df
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing data using various methods
        
        Args:
            df: DataFrame with missing data
            method: 'forward_fill', 'backward_fill', 'interpolate', 'drop'
        """
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in specified columns
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: 'iqr', 'zscore', 'modified_zscore'
            threshold: Threshold for outlier detection
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask[col] = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outlier_mask[col] = np.abs(modified_z_scores) > threshold
        
        # Add outlier information to DataFrame
        df_with_outliers = df.copy()
        df_with_outliers['is_outlier'] = outlier_mask.any(axis=1)
        df_with_outliers['outlier_columns'] = outlier_mask.apply(
            lambda row: ','.join(row[row].index), axis=1
        )
        
        return df_with_outliers
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df_tech = df.copy()
        
        if 'Close' in df.columns:
            # Simple Moving Averages
            for window in [5, 10, 20, 50, 100, 200]:
                df_tech[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            
            # Exponential Moving Averages
            for span in [5, 10, 20, 50]:
                df_tech[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
            
            # Returns
            df_tech['Returns'] = df['Close'].pct_change()
            df_tech['Log_Returns'] = np.log(df['Close']).diff()
            
            # Volatility measures
            for window in [10, 20, 50]:
                df_tech[f'Volatility_{window}'] = df_tech['Returns'].rolling(window).std() * np.sqrt(252)
        
        # RSI (Relative Strength Index)
        if 'Close' in df.columns:
            df_tech['RSI_14'] = self._calculate_rsi(df['Close'], 14)
            df_tech['RSI_30'] = self._calculate_rsi(df['Close'], 30)
        
        # MACD
        if 'Close' in df.columns:
            macd_data = self._calculate_macd(df['Close'])
            df_tech = pd.concat([df_tech, macd_data], axis=1)
        
        # Bollinger Bands
        if 'Close' in df.columns:
            bb_data = self._calculate_bollinger_bands(df['Close'], window=20, num_std=2)
            df_tech = pd.concat([df_tech, bb_data], axis=1)
        
        # Average True Range (ATR)
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            df_tech['ATR_14'] = self._calculate_atr(df[['High', 'Low', 'Close']], 14)
        
        # Volume indicators
        if all(col in df.columns for col in ['Close', 'Volume']):
            df_tech['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
            df_tech['Volume_Ratio'] = df['Volume'] / df_tech['Volume_SMA_20']
            df_tech['Price_Volume'] = df['Close'] * df['Volume']
        
        return df_tech
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicators
        
        Ensures that inputs/outputs are 1-dimensional Series even if a
        single-column DataFrame is passed in.
        """
        # Coerce prices to a 1D Series if a single-column DataFrame is provided
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] == 1:
                prices = prices.iloc[:, 0]
            else:
                # If multiple columns are provided, use the first by convention
                prices = prices.iloc[:, 0]

        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd_line = (ema_fast - ema_slow).squeeze()
        signal_line = macd_line.ewm(span=signal).mean().squeeze()
        histogram = (macd_line - signal_line).squeeze()

        # Ensure outputs are Series
        if isinstance(macd_line, pd.DataFrame):
            macd_line = macd_line.iloc[:, 0]
        if isinstance(signal_line, pd.DataFrame):
            signal_line = signal_line.iloc[:, 0]
        if isinstance(histogram, pd.DataFrame):
            histogram = histogram.iloc[:, 0]

        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        }, index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands
        
        Coerces a single-column DataFrame input to a Series to avoid 2D shape issues.
        """
        # Ensure 1D Series
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] == 1:
                prices = prices.iloc[:, 0]
            else:
                prices = prices.iloc[:, 0]

        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band,
            'BB_Width': upper_band - lower_band,
            'BB_Position': (prices - lower_band) / (upper_band - lower_band)
        }, index=prices.index)
    
    def _calculate_atr(self, ohlc_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = ohlc_data['High']
        low = ohlc_data['Low']
        close = ohlc_data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax', 
                      columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize data using various methods
        
        Args:
            df: Input DataFrame
            method: 'minmax', 'zscore', 'robust'
            columns: Columns to normalize
            
        Returns:
            Tuple of (normalized_df, normalization_params)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_norm = df.copy()
        norm_params = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                norm_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                df_norm[col] = (df[col] - mean_val) / std_val
                norm_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                
            elif method == 'robust':
                median_val = df[col].median()
                mad_val = np.median(np.abs(df[col] - median_val))
                df_norm[col] = (df[col] - median_val) / mad_val
                norm_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
        
        return df_norm, norm_params
    
    def denormalize_data(self, df_norm: pd.DataFrame, norm_params: Dict) -> pd.DataFrame:
        """Reverse normalization using stored parameters"""
        df_denorm = df_norm.copy()
        
        for col, params in norm_params.items():
            if col not in df_norm.columns:
                continue
                
            if params['method'] == 'minmax':
                df_denorm[col] = df_norm[col] * (params['max'] - params['min']) + params['min']
            elif params['method'] == 'zscore':
                df_denorm[col] = df_norm[col] * params['std'] + params['mean']
            elif params['method'] == 'robust':
                df_denorm[col] = df_norm[col] * params['mad'] + params['median']
        
        return df_denorm
    
    def create_features_for_ml(self, df: pd.DataFrame, target_col: str = 'Returns', 
                              lookback_periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create feature set for machine learning models
        
        Args:
            df: DataFrame with technical indicators
            target_col: Target variable column
            lookback_periods: Periods for creating lagged features
        """
        df_features = df.copy()
        
        # Lagged features
        feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
        
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                for lag in lookback_periods:
                    df_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics features
        if 'Close' in df.columns:
            for window in [5, 10, 20]:
                df_features[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window).mean()
                df_features[f'Close_rolling_std_{window}'] = df['Close'].rolling(window).std()
                df_features[f'Close_rolling_min_{window}'] = df['Close'].rolling(window).min()
                df_features[f'Close_rolling_max_{window}'] = df['Close'].rolling(window).max()
        
        # Interaction features
        if all(col in df.columns for col in ['Volume', 'Close']):
            df_features['Volume_Price_Interaction'] = df['Volume'] * df['Close']
        
        if all(col in df.columns for col in ['RSI_14', 'MACD']):
            df_features['RSI_MACD_Interaction'] = df['RSI_14'] * df['MACD']
        
        # Target variable (forward-looking returns)
        if target_col in df.columns:
            df_features['Target_1d'] = df[target_col].shift(-1)  # Next day return
            df_features['Target_5d'] = df[target_col].rolling(5).sum().shift(-5)  # 5-day forward return
        
        return df_features
    
    def save_to_database(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """Save DataFrame to SQLite database"""
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=True)
            print(f"Data saved to table '{table_name}' successfully")
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
    
    def load_from_database(self, table_name: str, start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """Load DataFrame from SQLite database"""
        try:
            query = f"SELECT * FROM {table_name}"
            
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append(f"Date >= '{start_date}'")
                if end_date:
                    conditions.append(f"Date <= '{end_date}'")
                query += " WHERE " + " AND ".join(conditions)
            
            df = pd.read_sql(query, self.engine, index_col='Date', parse_dates=['Date'])
            return df
        except Exception as e:
            print(f"Error loading from database: {str(e)}")
            return pd.DataFrame()
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        report = {
            'basic_info': {
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'date_range': {
                    'start': df.index.min() if hasattr(df.index, 'min') else None,
                    'end': df.index.max() if hasattr(df.index, 'max') else None
                },
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'missing_data': {},
            'data_types': df.dtypes.to_dict(),
            'outliers': {},
            'duplicates': df.duplicated().sum()
        }
        
        # Missing data analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        # Outlier analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outlier_df = self.detect_outliers(df[[col]], [col])
            outlier_count = outlier_df['is_outlier'].sum()
            report['outliers'][col] = {
                'count': outlier_count,
                'percentage': outlier_count / len(df) * 100
            }
        
        # Basic statistics for numeric columns
        report['statistics'] = df.describe().to_dict()
        
        return report
    
    def process_pipeline(self, raw_data: pd.DataFrame, ticker: str = "UNKNOWN") -> Dict:
        """
        Complete data processing pipeline
        
        Args:
            raw_data: Raw OHLCV data
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with processed data and metadata
        """
        print(f"Starting data processing pipeline for {ticker}")
        
        # Step 1: Clean price data
        print("Step 1: Cleaning price data...")
        cleaned_data = self.clean_price_data(raw_data)
        
        # Step 2: Handle missing data
        print("Step 2: Handling missing data...")
        filled_data = self.handle_missing_data(cleaned_data, method='forward_fill')
        
        # Step 3: Detect outliers
        print("Step 3: Detecting outliers...")
        outlier_data = self.detect_outliers(filled_data)
        
        # Step 4: Calculate technical indicators
        print("Step 4: Calculating technical indicators...")
        technical_data = self.calculate_technical_indicators(filled_data)
        
        # Step 5: Create ML features
        print("Step 5: Creating ML features...")
        feature_data = self.create_features_for_ml(technical_data)
        
        # Step 6: Generate quality report
        print("Step 6: Generating data quality report...")
        quality_report = self.generate_data_quality_report(feature_data)
        
        # Step 7: Save to database
        print("Step 7: Saving to database...")
        self.save_to_database(cleaned_data, f"{ticker}_raw")
        self.save_to_database(technical_data, f"{ticker}_technical")
        self.save_to_database(feature_data, f"{ticker}_features")
        
        print("Data processing pipeline completed successfully!")
        
        return {
            'raw_data': cleaned_data,
            'technical_data': technical_data,
            'feature_data': feature_data,
            'quality_report': quality_report,
            'outlier_info': outlier_data[['is_outlier', 'outlier_columns']],
            'ticker': ticker,
            'processing_date': datetime.now()
        }

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Initialize processor
    processor = DataProcessor()
    
    # Download sample data
    ticker = "AAPL"
    raw_data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    
    # Run complete processing pipeline
    results = processor.process_pipeline(raw_data, ticker)
    
    # Display results
    print(f"\n=== DATA PROCESSING RESULTS FOR {ticker} ===")
    print(f"Raw data shape: {results['raw_data'].shape}")
    print(f"Technical data shape: {results['technical_data'].shape}")
    print(f"Feature data shape: {results['feature_data'].shape}")
    
    print(f"\n=== DATA QUALITY REPORT ===")
    quality = results['quality_report']
    print(f"Date range: {quality['basic_info']['date_range']['start']} to {quality['basic_info']['date_range']['end']}")
    print(f"Total rows: {quality['basic_info']['num_rows']:,}")
    print(f"Total columns: {quality['basic_info']['num_columns']}")
    print(f"Memory usage: {quality['basic_info']['memory_usage_mb']:.2f} MB")
    print(f"Duplicate rows: {quality['duplicates']}")
    
    print(f"\n=== MISSING DATA SUMMARY ===")
    for col, info in quality['missing_data'].items():
        if info['count'] > 0:
            print(f"{col}: {info['count']} missing ({info['percentage']:.1f}%)")
    
    print(f"\n=== OUTLIERS SUMMARY ===")
    for col, info in quality['outliers'].items():
        if info['count'] > 0:
            print(f"{col}: {info['count']} outliers ({info['percentage']:.1f}%)")
    
    # Show sample of processed data
    print(f"\n=== SAMPLE TECHNICAL INDICATORS ===")
    tech_sample = results['technical_data'][['Close', 'SMA_20', 'RSI_14', 'MACD', 'BB_Upper', 'BB_Lower']].tail()
    print(tech_sample.round(2))