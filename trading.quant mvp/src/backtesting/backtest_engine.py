# src/backtesting/backtest_engine.py
import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradeRecord:
    """Individual trade record for tracking"""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: Optional[float]
    duration: Optional[int]  # days
    return_pct: Optional[float]

@dataclass
class PortfolioState:
    """Portfolio state at any point in time"""
    cash: float
    positions: Dict[str, int]  # symbol -> quantity
    portfolio_value: float
    date: datetime

class BacktestEngine:
    """
    Professional-grade backtesting engine with realistic trading costs and constraints
    
    Features:
    - Event-driven backtesting architecture
    - Realistic transaction costs and slippage
    - Position sizing and risk management
    - Detailed trade tracking and analytics
    - Multiple performance metrics
    - Monte Carlo validation
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,  # 0.1% per trade
                 slippage_rate: float = 0.0005,   # 0.05% slippage
                 borrowing_rate: float = 0.03,    # 3% annual interest for shorts
                 min_commission: float = 1.0):
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.borrowing_rate = borrowing_rate / 252  # Daily rate
        self.min_commission = min_commission
        
        # Initialize tracking variables
        self.reset()
    
    def reset(self):
        """Reset backtest state"""
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.portfolio_history = []
        self.current_date = None
        self.open_trades = {}  # symbol -> TradeRecord
        
    def calculate_transaction_cost(self, price: float, quantity: int) -> float:
        """Calculate realistic transaction costs"""
        trade_value = abs(price * quantity)
        commission = max(trade_value * self.commission_rate, self.min_commission)
        slippage = trade_value * self.slippage_rate
        return commission + slippage
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        position_value = sum(
            prices.get(symbol, 0) * quantity 
            for symbol, quantity in self.positions.items()
        )
        return self.cash + position_value
    
    def can_execute_trade(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if trade can be executed given current constraints"""
        trade_value = abs(quantity * price)
        transaction_cost = self.calculate_transaction_cost(price, quantity)
        
        if quantity > 0:  # Buy order
            required_cash = trade_value + transaction_cost
            return self.cash >= required_cash
        else:  # Sell order
            current_position = self.positions.get(symbol, 0)
            return current_position >= abs(quantity)
    
    def execute_trade(self, symbol: str, quantity: int, price: float, date: datetime) -> bool:
        """Execute a trade with realistic constraints"""
        if not self.can_execute_trade(symbol, quantity, price):
            return False
        
        # Calculate costs
        trade_value = quantity * price
        transaction_cost = self.calculate_transaction_cost(price, abs(quantity))
        
        # Update cash and positions
        self.cash -= (trade_value + transaction_cost)
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        # Handle trade records
        if quantity > 0:  # Opening/adding to position
            if symbol in self.open_trades:
                # Adding to existing position - close old trade and open new
                old_trade = self.open_trades[symbol]
                old_trade.exit_date = date
                old_trade.exit_price = price
                old_trade.pnl = (price - old_trade.entry_price) * old_trade.quantity
                old_trade.duration = (date - old_trade.entry_date).days
                old_trade.return_pct = old_trade.pnl / (old_trade.entry_price * old_trade.quantity)
                self.trades.append(old_trade)
            
            # Create new trade record
            trade_record = TradeRecord(
                entry_date=date,
                exit_date=None,
                symbol=symbol,
                side='long',
                entry_price=price,
                exit_price=None,
                quantity=self.positions[symbol],
                pnl=None,
                duration=None,
                return_pct=None
            )
            self.open_trades[symbol] = trade_record
            
        else:  # Closing/reducing position
            if symbol in self.open_trades:
                trade = self.open_trades[symbol]
                if self.positions[symbol] == 0:  # Fully closed
                    trade.exit_date = date
                    trade.exit_price = price
                    trade.pnl = (price - trade.entry_price) * trade.quantity
                    trade.duration = (date - trade.entry_date).days
                    trade.return_pct = trade.pnl / (trade.entry_price * trade.quantity)
                    self.trades.append(trade)
                    del self.open_trades[symbol]
        
        return True
    
    def run_backtest(self, price_data: pd.DataFrame, 
                    strategy_signals: pd.DataFrame,
                    position_sizer: Optional[Callable] = None) -> Dict:
        """
        Run complete backtest
        
        Args:
            price_data: DataFrame with OHLCV data
            strategy_signals: DataFrame with trading signals
            position_sizer: Function to determine position sizes
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        # Ensure data is aligned
        aligned_data = price_data.join(strategy_signals, how='inner')
        
        last_price = None
        symbol = 'ASSET'
        
        for date, row in aligned_data.iterrows():
            self.current_date = date
            
            # Extract price safely
            price = row['Close'] if 'Close' in row.index else (row.get('Adj Close') if 'Adj Close' in row.index else np.nan)
            if not pd.isna(price):
                last_price = price
            current_prices = {symbol: price} if not pd.isna(price) else {}
            
            # Extract signal safely
            signal = row['Signal'] if 'Signal' in row.index and not pd.isna(row['Signal']) else None
            
            if signal is not None and not pd.isna(price):
                current_position = self.positions.get(symbol, 0)
                
                if position_sizer:
                    # Custom sizer: delegate when an entry is desired, close fully on non-positive signal
                    if signal > 0 and current_position == 0:
                        target_quantity = position_sizer(signal, price, self.cash, self.calculate_portfolio_value(current_prices))
                    elif signal <= 0 and current_position > 0:
                        target_quantity = 0
                    else:
                        target_quantity = current_position
                else:
                    # Default: enter on positive signal if flat; exit fully on non-positive signal
                    if signal > 0 and current_position == 0:
                        target_value = self.cash * 0.1  # 10% of cash
                        target_quantity = int(target_value / price)
                    elif signal <= 0 and current_position > 0:
                        target_quantity = 0
                    else:
                        target_quantity = current_position
                
                trade_quantity = target_quantity - current_position
                if trade_quantity != 0:
                    self.execute_trade(symbol, trade_quantity, price, date)
            
            # Update portfolio history
            portfolio_value = self.calculate_portfolio_value(current_prices)
            portfolio_state = PortfolioState(
                cash=self.cash,
                positions=self.positions.copy(),
                portfolio_value=portfolio_value,
                date=date
            )
            self.portfolio_history.append(portfolio_state)
        
        # Close any remaining open trades
        # Close any remaining open trades at last known price
        if last_price is not None:
            final_prices = {symbol: last_price}
            for sym, trade in list(self.open_trades.items()):
                if sym in final_prices:
                    trade.exit_date = self.current_date
                    trade.exit_price = final_prices[sym]
                    trade.pnl = (final_prices[sym] - trade.entry_price) * trade.quantity
                    trade.duration = (self.current_date - trade.entry_date).days
                    trade.return_pct = trade.pnl / (trade.entry_price * trade.quantity)
                    self.trades.append(trade)
                    del self.open_trades[sym]
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history:
            return {}
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame([
            {
                'Date': state.date,
                'Portfolio_Value': state.portfolio_value,
                'Cash': state.cash
            }
            for state in self.portfolio_history
        ]).set_index('Date')
        
        # Calculate returns
        portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
        portfolio_df['Cumulative_Returns'] = (1 + portfolio_df['Returns']).cumprod()
        
        # Basic metrics
        total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / self.initial_capital) - 1
        
        # Risk metrics
        returns_series = portfolio_df['Returns'].dropna()
        
        if len(returns_series) > 0:
            volatility = returns_series.std() * np.sqrt(252)
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
            
            # Maximum drawdown
            rolling_max = portfolio_df['Portfolio_Value'].expanding().max()
            drawdown = (portfolio_df['Portfolio_Value'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = (returns_series.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            volatility = sharpe_ratio = max_drawdown = calmar_ratio = 0
        
        # Trade analysis
        trade_analysis = self.analyze_trades()
        
        # Exposure metrics
        portfolio_df['Exposure'] = (self.initial_capital - portfolio_df['Cash']) / self.initial_capital
        avg_exposure = portfolio_df['Exposure'].mean()
        
        # Build a clean trades dataframe for downstream rendering
        trades_df = pd.DataFrame([
            {
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'duration_days': t.duration,
            }
            for t in self.trades if t.exit_date is not None
        ])

        return {
            'total_return': total_return,
            'annualized_return': returns_series.mean() * 252 if len(returns_series) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'final_portfolio_value': portfolio_df['Portfolio_Value'].iloc[-1],
            'average_exposure': avg_exposure,
            'portfolio_history': portfolio_df,
            'trade_analysis': trade_analysis,
            'total_trades': len(self.trades),
            'trades': trades_df,
        }
    
    def analyze_trades(self) -> Dict:
        """Analyze individual trade performance"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_trade_duration': 0,
                # Ensure these keys always exist to avoid KeyError downstream
                'best_trade': 0,
                'worst_trade': 0
            }
        
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        
        total_trades = len(self.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        durations = [t.duration for t in self.trades if t.duration is not None]
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_duration,
            'best_trade': max([t.pnl for t in self.trades if t.pnl], default=0),
            'worst_trade': min([t.pnl for t in self.trades if t.pnl], default=0)
        }

# Example usage
if __name__ == "__main__":
    # Sample backtest
    import yfinance as yf
    from ..models.momentum_strategy import MomentumStrategy
    
    # Download data
    data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
    
    # Generate strategy signals
    strategy = MomentumStrategy()
    signals, _ = strategy.backtest(data)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    
    # Prepare data for backtest
    backtest_data = data[['Close']].copy()
    signal_data = signals[['Signal']].copy()
    
    results = engine.run_backtest(backtest_data, signal_data)
    
    # Print results
    print("=== BACKTEST RESULTS ===")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Volatility: {results['volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    
    print("\n=== TRADE ANALYSIS ===")
    trade_stats = results['trade_analysis']
    print(f"Total Trades: {trade_stats['total_trades']}")
    print(f"Win Rate: {trade_stats['win_rate']:.2%}")
    print(f"Profit Factor: {trade_stats['profit_factor']:.2f}")
    print(f"Average Trade Duration: {trade_stats['avg_trade_duration']:.1f} days")
    print(f"Best Trade: ${trade_stats['best_trade']:,.2f}")
    print(f"Worst Trade: ${trade_stats['worst_trade']:,.2f}")