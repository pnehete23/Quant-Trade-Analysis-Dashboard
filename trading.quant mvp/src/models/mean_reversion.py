# src/models/mean_reversion.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MeanReversionStrategy:
    """
    Advanced Mean Reversion Trading Strategy
    
    Features:
    - Bollinger Bands with dynamic parameters
    - Z-score based mean reversion signals
    - Cointegration-based pairs trading
    - Statistical arbitrage opportunities
    - Adaptive lookback periods
    - Risk management with position sizing
    """
    
    def __init__(self, 
                 lookback_window: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0,
                 min_half_life: int = 5,
                 max_half_life: int = 50,
                 max_position_size: float = 0.1):
        
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold  # Z-score threshold for entry
        self.exit_threshold = exit_threshold    # Z-score threshold for exit
        self.stop_loss_threshold = stop_loss_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.max_position_size = max_position_size
        
    def calculate_zscore(self, prices: pd.Series, window: int = None) -> pd.Series:
        """Calculate rolling Z-score for mean reversion signals"""
        if window is None:
            window = self.lookback_window
        
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        zscore = (prices - rolling_mean) / rolling_std
        return zscore
    
    def calculate_half_life(self, prices: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion using Ornstein-Uhlenbeck process
        
        The half-life indicates how long it takes for a price deviation to decay by half
        """
        try:
            # Calculate price differences
            price_diff = prices.diff().dropna()
            price_lag = prices.shift(1).dropna()
            
            # Align the series
            aligned_data = pd.DataFrame({
                'price_diff': price_diff,
                'price_lag': price_lag
            }).dropna()
            
            if len(aligned_data) < 10:
                return np.nan
            
            # Perform linear regression: Δp_t = α + β * p_{t-1} + ε_t
            X = aligned_data['price_lag'].values
            y = aligned_data['price_diff'].values
            
            # Add constant term
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # OLS regression
            coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            beta = coefficients[1]
            
            # Calculate half-life
            if beta >= 0:
                return np.nan  # No mean reversion
            
            half_life = -np.log(2) / beta
            
            # Validate half-life range
            if self.min_half_life <= half_life <= self.max_half_life:
                return half_life
            else:
                return np.nan
                
        except Exception as e:
            print(f"Error calculating half-life: {str(e)}")
            return np.nan
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = None, 
                                 num_std: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands for mean reversion analysis"""
        if window is None:
            window = self.lookback_window
        
        # Calculate moving average and standard deviation
        sma = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        # Calculate bands
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        
        # Band position (0 = lower band, 1 = upper band)
        band_position = (prices - lower_band) / (upper_band - lower_band)
        
        # Band width (volatility measure)
        band_width = (upper_band - lower_band) / sma
        
        return pd.DataFrame({
            'SMA': sma,
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'Band_Position': band_position,
            'Band_Width': band_width
        }, index=prices.index)
    
    def hurst_exponent(self, prices: pd.Series, lags: List[int] = None) -> float:
        """
        Calculate Hurst exponent to measure mean reversion tendency
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if lags is None:
            lags = range(2, min(100, len(prices) // 4))
        
        try:
            log_prices = np.log(prices.dropna())
            
            # Calculate variance of lagged differences
            variances = []
            for lag in lags:
                differences = log_prices.diff(lag).dropna()
                if len(differences) > 0:
                    variances.append(np.var(differences))
            
            if len(variances) < 3:
                return np.nan
            
            # Linear regression: log(variance) = log(c) + H * log(lag)
            log_lags = np.log(lags[:len(variances)])
            log_variances = np.log(variances)
            
            # Remove any infinite or NaN values
            valid_idx = np.isfinite(log_lags) & np.isfinite(log_variances)
            if np.sum(valid_idx) < 3:
                return np.nan
            
            # Perform regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_lags[valid_idx], log_variances[valid_idx]
            )
            
            # Hurst exponent is half the slope
            hurst = slope / 2.0
            
            return hurst if 0 < hurst < 1 else np.nan
            
        except Exception as e:
            print(f"Error calculating Hurst exponent: {str(e)}")
            return np.nan
    
    def adf_test(self, prices: pd.Series) -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity
        
        Returns test results including p-value and critical values
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Perform ADF test
            result = adfuller(prices.dropna(), autolag='AIC')
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05  # 5% significance level
            }
            
        except ImportError:
            print("statsmodels not available for ADF test")
            return {}
        except Exception as e:
            print(f"Error in ADF test: {str(e)}")
            return {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and indicators
        """
        df = data.copy()
        
        # Calculate Z-score
        df['Z_Score'] = self.calculate_zscore(df['Close'])
        
        # Calculate Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['Close'])
        df = pd.concat([df, bb_data], axis=1)
        
        # Calculate returns for analysis
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']).diff()
        
        # Calculate half-life (rolling)
        df['Half_Life'] = df['Close'].rolling(50).apply(
            lambda x: self.calculate_half_life(x) if len(x) == 50 else np.nan
        )
        
        # Calculate Hurst exponent (rolling)
        df['Hurst_Exponent'] = df['Close'].rolling(50).apply(
            lambda x: self.hurst_exponent(x) if len(x) == 50 else np.nan
        )
        
        # Generate trading signals
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        
        # Entry signals
        # Short when Z-score is high (price above mean)
        df.loc[df['Z_Score'] > self.entry_threshold, 'Signal'] = -1
        # Long when Z-score is low (price below mean)
        df.loc[df['Z_Score'] < -self.entry_threshold, 'Signal'] = 1
        
        # Exit signals (position closure)
        # Exit when Z-score returns toward mean
        df.loc[
            (abs(df['Z_Score']) < self.exit_threshold) & 
            (df['Signal'].shift(1) != 0), 
            'Signal'
        ] = 0
        
        # Stop-loss signals
        df.loc[abs(df['Z_Score']) > self.stop_loss_threshold, 'Signal'] = 0
        
        # Signal strength based on Z-score magnitude
        df['Signal_Strength'] = np.clip(abs(df['Z_Score']) / self.entry_threshold, 0, 2)
        
        # Position sizing based on signal strength and volatility
        volatility = df['Returns'].rolling(20).std()
        base_vol = 0.02  # 2% base volatility
        vol_adjustment = base_vol / volatility.fillna(base_vol)
        
        df['Position_Size'] = (
            df['Signal_Strength'] * 
            vol_adjustment * 
            self.max_position_size
        ).clip(0, self.max_position_size)
        
        # Filter signals based on mean reversion indicators
        # Only trade when Hurst < 0.5 (mean reverting)
        df.loc[df['Hurst_Exponent'] >= 0.5, 'Signal'] = 0
        
        # Only trade when half-life is reasonable
        df.loc[
            (df['Half_Life'] < self.min_half_life) | 
            (df['Half_Life'] > self.max_half_life), 
            'Signal'
        ] = 0
        
        # Track positions
        df['Position'] = df['Signal'].fillna(0)
        df['Position_Change'] = df['Position'].diff()
        
        return df
    
    def pairs_trading_signals(self, price1: pd.Series, price2: pd.Series, 
                             ticker1: str = "Asset1", ticker2: str = "Asset2") -> pd.DataFrame:
        """
        Generate pairs trading signals based on cointegration
        
        Args:
            price1: Price series for first asset
            price2: Price series for second asset
            ticker1: Name of first asset
            ticker2: Name of second asset
            
        Returns:
            DataFrame with pairs trading signals
        """
        # Align the series
        aligned_data = pd.DataFrame({
            ticker1: price1,
            ticker2: price2
        }).dropna()
        
        if len(aligned_data) < 50:
            raise ValueError("Insufficient data for pairs trading analysis")
        
        # Calculate hedge ratio using linear regression
        X = aligned_data[ticker2].values.reshape(-1, 1)
        y = aligned_data[ticker1].values
        
        # Add constant term
        X_with_const = np.column_stack([np.ones(len(X)), X.flatten()])
        coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        
        alpha = coefficients[0]  # Intercept
        beta = coefficients[1]   # Hedge ratio
        
        # Calculate spread
        spread = aligned_data[ticker1] - beta * aligned_data[ticker2] - alpha
        
        # Generate signals on the spread
        spread_df = pd.DataFrame({'Spread': spread}, index=aligned_data.index)
        
        # Apply mean reversion strategy to spread
        spread_zscore = self.calculate_zscore(spread)
        
        # Pairs trading signals
        signals_df = pd.DataFrame(index=aligned_data.index)
        signals_df['Spread'] = spread
        signals_df['Spread_ZScore'] = spread_zscore
        signals_df['Hedge_Ratio'] = beta
        
        # Trading signals
        signals_df['Pairs_Signal'] = 0
        
        # Long spread (buy asset1, sell asset2) when spread is low
        signals_df.loc[spread_zscore < -self.entry_threshold, 'Pairs_Signal'] = 1
        
        # Short spread (sell asset1, buy asset2) when spread is high  
        signals_df.loc[spread_zscore > self.entry_threshold, 'Pairs_Signal'] = -1
        
        # Exit when spread returns to mean
        signals_df.loc[abs(spread_zscore) < self.exit_threshold, 'Pairs_Signal'] = 0
        
        # Calculate individual asset signals
        signals_df[f'{ticker1}_Signal'] = signals_df['Pairs_Signal']
        signals_df[f'{ticker2}_Signal'] = -signals_df['Pairs_Signal'] * beta
        
        # Add cointegration test results
        coint_result = self.cointegration_test(aligned_data[ticker1], aligned_data[ticker2])
        for key, value in coint_result.items():
            signals_df[f'Coint_{key}'] = value
        
        return signals_df
    
    def cointegration_test(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Test for cointegration between two price series
        
        Returns:
            Dictionary with cointegration test results
        """
        try:
            from statsmodels.tsa.stattools import coint
            
            # Perform Engle-Granger cointegration test
            score, p_value, critical_values = coint(series1, series2)
            
            return {
                'score': score,
                'p_value': p_value,
                'critical_values_1pct': critical_values[0],
                'critical_values_5pct': critical_values[1],
                'critical_values_10pct': critical_values[2],
                'is_cointegrated': p_value < 0.05
            }
            
        except ImportError:
            print("statsmodels not available for cointegration test")
            return {}
        except Exception as e:
            print(f"Error in cointegration test: {str(e)}")
            return {}
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Tuple[pd.DataFrame, Dict]:
        """
        Backtest the mean reversion strategy
        
        Args:
            data: Historical price data
            initial_capital: Starting capital
            
        Returns:
            Tuple of (signals_dataframe, performance_metrics)
        """
        # Generate signals
        signals_df = self.generate_signals(data)
        
        # Calculate strategy returns
        signals_df['Strategy_Returns'] = (
            signals_df['Position'].shift(1) * 
            signals_df['Returns'] * 
            signals_df['Position_Size'].shift(1)
        )
        
        # Calculate cumulative returns
        signals_df['Cumulative_Returns'] = (1 + signals_df['Strategy_Returns']).cumprod()
        signals_df['Benchmark_Returns'] = (1 + signals_df['Returns']).cumprod()
        
        # Performance metrics
        strategy_returns = signals_df['Strategy_Returns'].dropna()
        benchmark_returns = signals_df['Returns'].dropna()
        
        total_return = signals_df['Cumulative_Returns'].iloc[-1] - 1
        benchmark_return = signals_df['Benchmark_Returns'].iloc[-1] - 1
        
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # Max drawdown
        rolling_max = signals_df['Cumulative_Returns'].expanding().max()
        drawdown = (signals_df['Cumulative_Returns'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (strategy_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Mean reversion specific metrics
        avg_half_life = signals_df['Half_Life'].mean()
        avg_hurst = signals_df['Hurst_Exponent'].mean()
        
        performance = {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'alpha': total_return - benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_half_life': avg_half_life,
            'avg_hurst_exponent': avg_hurst
        }
        
        return signals_df, performance

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Download sample data
    ticker = "GLD"  # Gold ETF (good for mean reversion)
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    
    # Initialize strategy
    strategy = MeanReversionStrategy(
        lookback_window=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        max_position_size=0.15
    )
    
    # Run backtest
    signals, performance = strategy.backtest(data)
    
    # Print results
    print(f"=== {ticker} Mean Reversion Strategy Backtest ===")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Benchmark Return: {performance['benchmark_return']:.2%}")
    print(f"Alpha: {performance['alpha']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Average Half-Life: {performance['avg_half_life']:.1f} days")
    print(f"Average Hurst Exponent: {performance['avg_hurst_exponent']:.3f}")
    
    # Show recent signals
    print("\n=== Recent Signals ===")
    recent_signals = signals[['Close', 'Z_Score', 'Signal', 'Position_Size', 'Half_Life']].tail(10)
    print(recent_signals.round(3))
    
    # Test pairs trading
    print("\n=== Pairs Trading Example ===")
    spy_data = yf.download("SPY", start="2020-01-01", end="2024-01-01")
    pairs_signals = strategy.pairs_trading_signals(
        data['Close'], spy_data['Close'], "GLD", "SPY"
    )
    print(f"Pairs trading signals shape: {pairs_signals.shape}")
    print(f"Cointegration p-value: {pairs_signals['Coint_p_value'].iloc[-1]:.4f}")
    print(f"Hedge ratio: {pairs_signals['Hedge_Ratio'].iloc[-1]:.3f}")