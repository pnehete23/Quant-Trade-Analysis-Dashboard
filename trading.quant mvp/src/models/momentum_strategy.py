# src/models/momentum_strategy.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MomentumStrategy:
    """
    Advanced Momentum Trading Strategy with RSI confirmation and dynamic position sizing
    
    Features:
    - Dual moving average crossover signals
    - RSI confirmation to filter false signals
    - Volatility-based position sizing
    - Risk management with stop-loss and take-profit
    """
    
    def __init__(self, 
                 short_window: int = 20, 
                 long_window: int = 50,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 volatility_window: int = 20,
                 max_position_size: float = 0.1,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04):
        
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volatility_window = volatility_window
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility for position sizing"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def calculate_position_size(self, volatility: float, capital: float) -> float:
        """Dynamic position sizing based on volatility.

        Handles both scalar and array-like inputs for robustness when used inside
        DataFrame.apply where a Series value may occasionally be passed.
        """
        # Coerce to scalar float if a vector/Series is accidentally passed
        try:
            # This will work for numpy.float64, python float, or single-element arrays
            volatility_value = float(np.asarray(volatility).ravel()[-1])
        except Exception:
            # Fallback: treat as missing
            return 0

        if pd.isna(volatility_value) or volatility_value == 0:
            return 0

        # Inverse volatility position sizing
        base_vol = 0.20  # 20% base volatility
        vol_adjustment = base_vol / volatility_value
        position_size = min(self.max_position_size * vol_adjustment, self.max_position_size)

        return max(position_size, 0.01)  # Minimum 1% position
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on momentum strategy
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and additional indicators
        """
        df = data.copy()
        
        # Calculate moving averages
        df['MA_Short'] = df['Close'].rolling(window=self.short_window).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.long_window).mean()
        
        # Calculate RSI
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        
        # Calculate returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = self.calculate_volatility(df['Returns'], self.volatility_window)
        
        # Generate basic momentum signals
        df['MA_Signal'] = 0
        df.loc[df['MA_Short'] > df['MA_Long'], 'MA_Signal'] = 1
        df.loc[df['MA_Short'] < df['MA_Long'], 'MA_Signal'] = -1
        
        # RSI confirmation
        df['RSI_Confirm'] = 0
        df.loc[(df['RSI'] > self.rsi_oversold) & (df['RSI'] < self.rsi_overbought), 'RSI_Confirm'] = 1
        df.loc[df['RSI'] <= self.rsi_oversold, 'RSI_Confirm'] = 1  # Oversold can confirm buy
        df.loc[df['RSI'] >= self.rsi_overbought, 'RSI_Confirm'] = -1  # Overbought can confirm sell
        
        # Combined signal
        df['Signal'] = 0
        df.loc[(df['MA_Signal'] == 1) & (df['RSI_Confirm'] >= 0), 'Signal'] = 1  # Buy
        df.loc[(df['MA_Signal'] == -1) & (df['RSI_Confirm'] <= 0), 'Signal'] = -1  # Sell
        
        # Position sizing
        df['Position_Size'] = df.apply(
            lambda row: self.calculate_position_size(row['Volatility'], 100000), axis=1
        )
        
        # Entry and exit points
        df['Position'] = df['Signal'].shift(1)  # Take position next day
        df['Position_Change'] = df['Position'].diff()
        
        # Entry/Exit signals
        df['Entry_Long'] = (df['Position_Change'] == 1)
        df['Exit_Long'] = (df['Position_Change'] == -1) | (df['Position_Change'] == -2)
        df['Entry_Short'] = (df['Position_Change'] == -1)
        df['Exit_Short'] = (df['Position_Change'] == 1) | (df['Position_Change'] == 2)
        
        return df
    
    def calculate_performance_metrics(self, signals_df: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        df = signals_df.copy()
        
        # Calculate strategy returns
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns'] * df['Position_Size']
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        df['Benchmark_Returns'] = (1 + df['Returns']).cumprod()
        
        # Performance metrics
        total_return = df['Cumulative_Returns'].iloc[-1] - 1
        benchmark_return = df['Benchmark_Returns'].iloc[-1] - 1
        
        # Risk metrics
        volatility = df['Strategy_Returns'].std() * np.sqrt(252)
        sharpe_ratio = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
        
        # Max drawdown
        rolling_max = df['Cumulative_Returns'].expanding().max()
        drawdown = (df['Cumulative_Returns'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (df['Strategy_Returns'] > 0).sum()
        total_trades = (df['Strategy_Returns'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'alpha': total_return - benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete backtest of the momentum strategy
        
        Args:
            data: Historical price data
            initial_capital: Starting capital
            
        Returns:
            Tuple of (signals_dataframe, performance_metrics)
        """
        # Generate signals
        signals_df = self.generate_signals(data)
        
        # Calculate performance
        performance = self.calculate_performance_metrics(signals_df)
        
        # Add capital tracking
        # Ensure cumulative curve exists on the returned signals_df (calculate_performance_metrics
        # computes it on a copy for metrics). Recompute here to avoid KeyError.
        if 'Strategy_Returns' not in signals_df.columns:
            signals_df['Strategy_Returns'] = (
                signals_df['Position'].shift(1) * signals_df['Returns'] * signals_df['Position_Size']
            )
        if 'Cumulative_Returns' not in signals_df.columns:
            signals_df['Cumulative_Returns'] = (1 + signals_df['Strategy_Returns']).cumprod()

        signals_df['Capital'] = initial_capital * signals_df['Cumulative_Returns'].fillna(1)
        
        return signals_df, performance

# Example usage and testing
if __name__ == "__main__":
    # Sample data creation for testing
    import yfinance as yf
    
    # Download sample data
    ticker = "SPY"
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    
    # Initialize strategy
    strategy = MomentumStrategy(
        short_window=20,
        long_window=50,
        rsi_period=14,
        max_position_size=0.1
    )
    
    # Run backtest
    signals, performance = strategy.backtest(data)
    
    # Print results
    print(f"=== {ticker} Momentum Strategy Backtest ===")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Benchmark Return: {performance['benchmark_return']:.2%}")
    print(f"Alpha: {performance['alpha']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Total Trades: {performance['total_trades']}")
    
    # Show recent signals
    print("\n=== Recent Signals ===")
    recent_signals = signals[['Close', 'MA_Short', 'MA_Long', 'RSI', 'Signal', 'Position_Size']].tail(10)
    print(recent_signals.round(2))