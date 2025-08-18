# src/risk_management/portfolio_risk.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class PortfolioRiskManager:
    """
    Comprehensive Portfolio Risk Management System
    
    Features:
    - Value at Risk (VaR) calculation using multiple methods
    - Conditional Value at Risk (CVaR)
    - Portfolio correlation analysis
    - Stress testing and scenario analysis
    - Risk decomposition and attribution
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.portfolio_data = None
        self.returns_data = None
    
    def load_portfolio_data(self, returns_df: pd.DataFrame, weights: Optional[Dict] = None):
        """Load portfolio returns data and weights"""
        self.returns_data = returns_df
        
        if weights is None:
            # Equal weight if not specified
            n_assets = len(returns_df.columns)
            weights = {col: 1/n_assets for col in returns_df.columns}
        
        self.weights = weights
        self.portfolio_returns = self._calculate_portfolio_returns(returns_df, weights)
    
    def _calculate_portfolio_returns(self, returns_df: pd.DataFrame, weights: Dict) -> pd.Series:
        """Calculate portfolio-level returns"""
        weighted_returns = returns_df * pd.Series(weights)
        return weighted_returns.sum(axis=1)
    
    def historical_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk using Historical Simulation method
        
        Args:
            returns: Portfolio returns series
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            VaR value (positive number representing loss)
        """
        if len(returns.dropna()) == 0:
            return np.nan
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        # Calculate percentile (VaR is the negative of the percentile)
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(clean_returns, var_percentile)
        
        return -var_value  # Return as positive number (loss)
    
    def parametric_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate VaR using Parametric (Normal) method
        
        Args:
            returns: Portfolio returns series
            confidence_level: Confidence level
            
        Returns:
            VaR value
        """
        if len(returns.dropna()) == 0:
            return np.nan
        
        clean_returns = returns.dropna()
        mean_return = clean_returns.mean()
        std_return = clean_returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var_value = mean_return + z_score * std_return
        
        return -var_value
    
    def monte_carlo_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                       n_simulations: int = 10000) -> float:
        """
        Calculate VaR using Monte Carlo simulation
        
        Args:
            returns: Portfolio returns series
            confidence_level: Confidence level
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            VaR value
        """
        if len(returns.dropna()) == 0:
            return np.nan
        
        clean_returns = returns.dropna()
        mean_return = clean_returns.mean()
        std_return = clean_returns.std()
        
        # Generate random scenarios
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(simulated_returns, var_percentile)
        
        return -var_value
    
    def conditional_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Portfolio returns series
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        if len(returns.dropna()) == 0:
            return np.nan
        
        clean_returns = returns.dropna()
        var_threshold = -self.historical_var(clean_returns, confidence_level)
        
        # Calculate expected value of returns below VaR threshold
        tail_returns = clean_returns[clean_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return self.historical_var(clean_returns, confidence_level)
        
        cvar_value = tail_returns.mean()
        
        return -cvar_value
    
    def calculate_var_summary(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate VaR using all methods for comparison"""
        results = []
        
        for confidence_level in self.confidence_levels:
            historical = self.historical_var(returns, confidence_level)
            parametric = self.parametric_var(returns, confidence_level)
            monte_carlo = self.monte_carlo_var(returns, confidence_level)
            conditional = self.conditional_var(returns, confidence_level)
            
            results.append({
                'Confidence_Level': f"{confidence_level:.0%}",
                'Historical_VaR': historical,
                'Parametric_VaR': parametric,
                'Monte_Carlo_VaR': monte_carlo,
                'Conditional_VaR': conditional
            })
        
        return pd.DataFrame(results)
    
    def correlation_analysis(self, returns_df: pd.DataFrame) -> Dict:
        """Analyze correlations between assets"""
        corr_matrix = returns_df.corr()
        
        # Calculate average correlation
        n = len(corr_matrix)
        avg_correlation = (corr_matrix.sum().sum() - n) / (n * (n - 1))
        
        # Find highest and lowest correlations
        corr_values = corr_matrix.values
        np.fill_diagonal(corr_values, np.nan)  # Ignore diagonal
        
        max_corr = np.nanmax(corr_values)
        min_corr = np.nanmin(corr_values)
        
        # Find the pairs
        max_idx = np.unravel_index(np.nanargmax(corr_values), corr_values.shape)
        min_idx = np.unravel_index(np.nanargmin(corr_values), corr_values.shape)
        
        max_pair = (corr_matrix.index[max_idx[0]], corr_matrix.columns[max_idx[1]])
        min_pair = (corr_matrix.index[min_idx[0]], corr_matrix.columns[min_idx[1]])
        
        return {
            'correlation_matrix': corr_matrix,
            'average_correlation': avg_correlation,
            'max_correlation': max_corr,
            'min_correlation': min_corr,
            'max_corr_pair': max_pair,
            'min_corr_pair': min_pair
        }
    
    def stress_testing(self, returns_df: pd.DataFrame, scenarios: Dict) -> pd.DataFrame:
        """
        Perform stress testing under various market scenarios
        
        Args:
            returns_df: Asset returns data
            scenarios: Dictionary of stress scenarios
                      e.g., {'market_crash': -0.20, 'volatility_spike': 2.0}
        """
        results = []
        
        for scenario_name, scenario_params in scenarios.items():
            if scenario_name == 'market_crash':
                # Apply uniform shock to all assets
                shocked_returns = returns_df + scenario_params
                portfolio_impact = self._calculate_portfolio_returns(shocked_returns, self.weights).sum()
                
            elif scenario_name == 'volatility_spike':
                # Increase volatility by factor
                mean_returns = returns_df.mean()
                shocked_returns = returns_df * scenario_params + mean_returns * (1 - scenario_params)
                portfolio_impact = self._calculate_portfolio_returns(shocked_returns, self.weights).sum()
                
            elif scenario_name == 'correlation_breakdown':
                # Increase all correlations to scenario_params value
                shocked_returns = self._simulate_correlation_shock(returns_df, scenario_params)
                portfolio_impact = self._calculate_portfolio_returns(shocked_returns, self.weights).sum()
            
            else:
                portfolio_impact = 0
            
            results.append({
                'Scenario': scenario_name,
                'Portfolio_Impact': portfolio_impact,
                'Impact_Percentage': portfolio_impact * 100
            })
        
        return pd.DataFrame(results)
    
    def _simulate_correlation_shock(self, returns_df: pd.DataFrame, target_corr: float) -> pd.DataFrame:
        """Simulate a correlation shock scenario"""
        # This is a simplified correlation shock - in practice, you'd use more sophisticated methods
        mean_returns = returns_df.mean()
        std_returns = returns_df.std()
        
        # Generate correlated random returns
        n_assets = len(returns_df.columns)
        n_periods = len(returns_df)
        
        # Create correlation matrix with target correlation
        corr_matrix = np.full((n_assets, n_assets), target_corr)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Generate correlated random variables
        random_returns = np.random.multivariate_normal(
            mean_returns.values, 
            np.outer(std_returns.values, std_returns.values) * corr_matrix, 
            n_periods
        )
        
        return pd.DataFrame(random_returns, columns=returns_df.columns, index=returns_df.index)
    
    def risk_decomposition(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Decompose portfolio risk by asset contribution"""
        portfolio_var = self.portfolio_returns.var()
        
        contributions = []
        for asset in returns_df.columns:
            # Calculate marginal contribution to risk
            asset_weight = self.weights[asset]
            asset_returns = returns_df[asset]
            
            # Covariance with portfolio
            covariance = np.cov(asset_returns.dropna(), self.portfolio_returns.dropna())[0, 1]
            
            # Risk contribution
            marginal_contrib = covariance / portfolio_var if portfolio_var != 0 else 0
            total_contrib = asset_weight * marginal_contrib
            
            contributions.append({
                'Asset': asset,
                'Weight': asset_weight,
                'Marginal_Risk_Contrib': marginal_contrib,
                'Total_Risk_Contrib': total_contrib,
                'Risk_Contrib_Percentage': total_contrib * 100
            })
        
        return pd.DataFrame(contributions)
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        if self.returns_data is None or self.portfolio_returns is None:
            raise ValueError("Portfolio data not loaded. Call load_portfolio_data() first.")
        
        # VaR Analysis
        var_summary = self.calculate_var_summary(self.portfolio_returns)
        
        # Correlation Analysis
        correlation_analysis = self.correlation_analysis(self.returns_data)
        
        # Stress Testing
        stress_scenarios = {
            'market_crash': -0.10,  # 10% market crash
            'volatility_spike': 2.0,  # Double volatility
            'correlation_breakdown': 0.8  # High correlation regime
        }
        stress_results = self.stress_testing(self.returns_data, stress_scenarios)
        
        # Risk Decomposition
        risk_decomp = self.risk_decomposition(self.returns_data)
        
        # Portfolio Statistics
        portfolio_stats = {
            'daily_return_mean': self.portfolio_returns.mean(),
            'daily_return_std': self.portfolio_returns.std(),
            'annualized_return': self.portfolio_returns.mean() * 252,
            'annualized_volatility': self.portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (self.portfolio_returns.mean() / self.portfolio_returns.std()) * np.sqrt(252),
            'skewness': stats.skew(self.portfolio_returns.dropna()),
            'kurtosis': stats.kurtosis(self.portfolio_returns.dropna())
        }
        
        return {
            'var_analysis': var_summary,
            'correlation_analysis': correlation_analysis,
            'stress_testing': stress_results,
            'risk_decomposition': risk_decomp,
            'portfolio_statistics': portfolio_stats
        }

# Example usage
if __name__ == "__main__":
    # Sample usage with synthetic data
    np.random.seed(42)
    
    # Generate sample returns data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_assets = 5
    n_periods = len(dates)
    
    # Create correlated asset returns
    base_returns = np.random.normal(0.0005, 0.015, (n_periods, n_assets))  # Daily returns
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
    
    returns_df = pd.DataFrame(base_returns, index=dates, columns=asset_names)
    
    # Portfolio weights
    weights = {'AAPL': 0.2, 'GOOGL': 0.2, 'MSFT': 0.2, 'TSLA': 0.2, 'SPY': 0.2}
    
    # Initialize risk manager
    risk_manager = PortfolioRiskManager()
    risk_manager.load_portfolio_data(returns_df, weights)
    
    # Generate comprehensive risk report
    risk_report = risk_manager.generate_risk_report()
    
    # Display results
    print("=== PORTFOLIO RISK ANALYSIS REPORT ===\n")
    
    print("1. VALUE AT RISK ANALYSIS")
    print(risk_report['var_analysis'].round(4))
    print()
    
    print("2. PORTFOLIO STATISTICS")
    for key, value in risk_report['portfolio_statistics'].items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print()
    
    print("3. CORRELATION ANALYSIS")
    print(f"Average Correlation: {risk_report['correlation_analysis']['average_correlation']:.3f}")
    print(f"Highest Correlation: {risk_report['correlation_analysis']['max_correlation']:.3f} "
          f"({risk_report['correlation_analysis']['max_corr_pair']})")
    print(f"Lowest Correlation: {risk_report['correlation_analysis']['min_correlation']:.3f} "
          f"({risk_report['correlation_analysis']['min_corr_pair']})")
    print()
    
    print("4. STRESS TESTING RESULTS")
    print(risk_report['stress_testing'].round(4))
    print()
    
    print("5. RISK DECOMPOSITION")
    print(risk_report['risk_decomposition'].round(4))