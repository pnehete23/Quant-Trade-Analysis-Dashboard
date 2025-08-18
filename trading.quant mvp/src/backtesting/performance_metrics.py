# src/backtesting/performance_metrics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """
    Comprehensive performance metrics calculator for trading strategies
    
    Features:
    - Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
    - Drawdown analysis and recovery periods
    - Rolling performance windows
    - Benchmark comparison and attribution
    - Statistical significance testing
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
    def calculate_returns_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Calculate comprehensive return-based metrics"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return self._empty_metrics()
        
        # Basic return metrics
        total_return = (1 + clean_returns).prod() - 1
        annualized_return = (1 + clean_returns.mean()) ** 252 - 1
        annualized_volatility = clean_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = clean_returns - self.risk_free_rate
        sharpe_ratio = excess_returns.mean() / clean_returns.std() * np.sqrt(252) if clean_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = clean_returns[clean_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / (downside_returns.std() if len(downside_returns) > 0 else 1) * np.sqrt(252)
        
        # Skewness and Kurtosis
        skewness = stats.skew(clean_returns)
        kurtosis = stats.kurtosis(clean_returns)
        
        # VaR and CVaR
        var_95 = np.percentile(clean_returns, 5)
        var_99 = np.percentile(clean_returns, 1)
        cvar_95 = clean_returns[clean_returns <= var_95].mean() if len(clean_returns[clean_returns <= var_95]) > 0 else var_95
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation
        }
        
        # Benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(clean_returns, benchmark_returns)
            metrics.update(benchmark_metrics)
        
        return metrics
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Calculate benchmark comparison metrics"""
        # Align returns
        aligned_data = pd.DataFrame({'strategy': returns, 'benchmark': benchmark_returns}).dropna()
        
        if len(aligned_data) == 0:
            return {}
        
        strategy_returns = aligned_data['strategy']
        bench_returns = aligned_data['benchmark']
        
        # Alpha and Beta calculation
        excess_strategy = strategy_returns - self.risk_free_rate
        excess_benchmark = bench_returns - self.risk_free_rate
        
        if excess_benchmark.std() > 0:
            beta = np.cov(excess_strategy, excess_benchmark)[0, 1] / excess_benchmark.var()
            alpha = excess_strategy.mean() - beta * excess_benchmark.mean()
        else:
            beta = 0
            alpha = excess_strategy.mean()
        
        # Tracking error and information ratio
        active_returns = strategy_returns - bench_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
        
        # Up/Down capture ratios
        up_market = bench_returns > 0
        down_market = bench_returns < 0
        
        up_capture = (strategy_returns[up_market].mean() / bench_returns[up_market].mean()) if bench_returns[up_market].mean() != 0 else 0
        down_capture = (strategy_returns[down_market].mean() / bench_returns[down_market].mean()) if bench_returns[down_market].mean() != 0 else 0
        
        return {
            'alpha': alpha * 252,  # Annualized
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'correlation': strategy_returns.corr(bench_returns)
        }
    
    def calculate_drawdown_metrics(self, cumulative_returns: pd.Series) -> Dict:
        """Calculate comprehensive drawdown analysis"""
        if len(cumulative_returns) == 0:
            return {}
        
        # Calculate running maximum and drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Drawdown duration analysis
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        
        if drawdown_periods:
            avg_drawdown_duration = np.mean([period['duration'] for period in drawdown_periods])
            max_drawdown_duration = max([period['duration'] for period in drawdown_periods])
            recovery_periods = [period['recovery_days'] for period in drawdown_periods if period['recovery_days'] is not None]
            avg_recovery_time = np.mean(recovery_periods) if recovery_periods else None
        else:
            avg_drawdown_duration = 0
            max_drawdown_duration = 0
            avg_recovery_time = None
        
        # Calmar ratio (annualized return / max drawdown)
        annual_return = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]) ** (252 / len(cumulative_returns)) - 1
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_dd_date,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_recovery_time': avg_recovery_time,
            'calmar_ratio': calmar_ratio,
            'current_drawdown': drawdown.iloc[-1],
            'drawdown_periods': drawdown_periods
        }
    
    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[Dict]:
        """Identify individual drawdown periods and their characteristics"""
        periods = []
        in_drawdown = False
        start_date = None
        peak_date = None
        
        for date, dd_value in drawdown.items():
            if dd_value < 0 and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                start_date = date
                peak_date = date
                min_dd = dd_value
                
            elif dd_value < 0 and in_drawdown:
                # Continuing drawdown
                if dd_value < min_dd:
                    min_dd = dd_value
                    peak_date = date
                    
            elif dd_value >= 0 and in_drawdown:
                # End of drawdown (recovery)
                duration = (peak_date - start_date).days
                recovery_days = (date - peak_date).days
                
                periods.append({
                    'start_date': start_date,
                    'peak_date': peak_date,
                    'end_date': date,
                    'duration': duration,
                    'recovery_days': recovery_days,
                    'max_drawdown': min_dd
                })
                
                in_drawdown = False
        
        # Handle case where strategy is still in drawdown
        if in_drawdown:
            duration = (peak_date - start_date).days
            periods.append({
                'start_date': start_date,
                'peak_date': peak_date,
                'end_date': drawdown.index[-1],
                'duration': duration,
                'recovery_days': None,  # Still in drawdown
                'max_drawdown': min_dd
            })
        
        return periods
    
    def calculate_rolling_metrics(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=returns.index[window-1:])
        
        # Rolling returns and volatility
        rolling_metrics['rolling_return'] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        rolling_metrics['rolling_sharpe'] = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )
        
        # Rolling maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window).max()
        rolling_dd = (cumulative_returns - rolling_max) / rolling_max
        rolling_metrics['rolling_max_drawdown'] = rolling_dd.rolling(window).min()
        
        return rolling_metrics
    
    def calculate_monthly_metrics(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate monthly performance breakdown"""
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        monthly_stats = pd.DataFrame({
            'Monthly_Return': monthly_returns,
            'Positive_Months': monthly_returns > 0,
            'Month': [f"{year}-{month:02d}" for year, month in monthly_returns.index]
        })
        
        # Monthly statistics
        monthly_summary = {
            'avg_monthly_return': monthly_returns.mean(),
            'monthly_volatility': monthly_returns.std(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months_pct': (monthly_returns > 0).mean(),
            'monthly_sharpe': monthly_returns.mean() / monthly_returns.std() * np.sqrt(12) if monthly_returns.std() > 0 else 0
        }
        
        return monthly_stats, monthly_summary
    
    def statistical_significance_test(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Test statistical significance of performance"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:  # Minimum sample size
            return {'error': 'Insufficient data for statistical testing'}
        
        # T-test for mean return significantly different from zero
        t_stat, p_value = stats.ttest_1samp(clean_returns, 0)
        
        results = {
            'sample_size': len(clean_returns),
            'mean_return_ttest': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_5pct': p_value < 0.05
            }
        }
        
        # Test against benchmark if provided
        if benchmark_returns is not None:
            aligned_data = pd.DataFrame({'strategy': returns, 'benchmark': benchmark_returns}).dropna()
            if len(aligned_data) > 30:
                excess_returns = aligned_data['strategy'] - aligned_data['benchmark']
                t_stat_excess, p_value_excess = stats.ttest_1samp(excess_returns, 0)
                
                results['excess_return_ttest'] = {
                    't_statistic': t_stat_excess,
                    'p_value': p_value_excess,
                    'significant_at_5pct': p_value_excess < 0.05
                }
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(clean_returns)
        results['normality_test'] = {
            'jarque_bera_statistic': jb_stat,
            'p_value': jb_pvalue,
            'normally_distributed': jb_pvalue > 0.05
        }
        
        return results
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict when no data available"""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'skewness': 0,
            'kurtosis': 0,
            'var_95': 0,
            'var_99': 0,
            'cvar_95': 0
        }
    
    def generate_performance_report(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Generate comprehensive performance report"""
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Basic metrics
        return_metrics = self.calculate_returns_metrics(returns, benchmark_returns)
        
        # Drawdown analysis
        drawdown_metrics = self.calculate_drawdown_metrics(cumulative_returns)
        
        # Rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(returns)
        
        # Monthly breakdown
        monthly_stats, monthly_summary = self.calculate_monthly_metrics(returns)
        
        # Statistical tests
        significance_tests = self.statistical_significance_test(returns, benchmark_returns)
        
        return {
            'return_metrics': return_metrics,
            'drawdown_metrics': drawdown_metrics,
            'rolling_metrics': rolling_metrics,
            'monthly_stats': monthly_stats,
            'monthly_summary': monthly_summary,
            'significance_tests': significance_tests,
            'cumulative_returns': cumulative_returns
        }

# Example usage
if __name__ == "__main__":
    # Sample usage with synthetic data
    np.random.seed(42)
    
    # Generate sample returns
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    strategy_returns = pd.Series(
        np.random.normal(0.0008, 0.015, len(dates)),  # 8bps daily mean, 1.5% daily vol
        index=dates
    )
    
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(dates)),  # 5bps daily mean, 1.2% daily vol  
        index=dates
    )
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # Generate comprehensive report
    report = analyzer.generate_performance_report(strategy_returns, benchmark_returns)
    
    # Display key results
    print("=== PERFORMANCE ANALYSIS REPORT ===\n")
    
    print("1. RETURN METRICS")
    for key, value in report['return_metrics'].items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print()
    
    print("2. DRAWDOWN METRICS")
    dd_metrics = report['drawdown_metrics']
    print(f"Max Drawdown: {dd_metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {dd_metrics['calmar_ratio']:.2f}")
    print(f"Avg Recovery Time: {dd_metrics['avg_recovery_time']:.0f} days" if dd_metrics['avg_recovery_time'] else "N/A")
    print()
    
    print("3. MONTHLY SUMMARY")
    monthly_sum = report['monthly_summary']
    for key, value in monthly_sum.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print()
    
    print("4. STATISTICAL SIGNIFICANCE")
    sig_tests = report['significance_tests']
    if 'mean_return_ttest' in sig_tests:
        print(f"Mean Return T-Test P-Value: {sig_tests['mean_return_ttest']['p_value']:.4f}")
        print(f"Statistically Significant: {sig_tests['mean_return_ttest']['significant_at_5pct']}")
    
    if 'excess_return_ttest' in sig_tests:
        print(f"Excess Return P-Value: {sig_tests['excess_return_ttest']['p_value']:.4f}")
        print(f"Outperforms Benchmark: {sig_tests['excess_return_ttest']['significant_at_5pct']}")