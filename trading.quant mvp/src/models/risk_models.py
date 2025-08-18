# src/models/risk_models.py
import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import optimize
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

class RiskModels:
    """
    Advanced Risk Modeling Suite for Portfolio Management
    
    Features:
    - GARCH volatility modeling
    - Factor risk models (Fama-French, custom factors)
    - Extreme value theory for tail risk
    - Copula-based dependency modeling
    - Dynamic correlation models
    - Risk budgeting and allocation
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        
    def garch_volatility_forecast(self, returns: pd.Series, horizon: int = 1) -> Dict:
        """
        GARCH(1,1) volatility forecasting
        
        Args:
            returns: Return series
            horizon: Forecast horizon in days
            
        Returns:
            Dictionary with GARCH parameters and forecasts
        """
        try:
            from arch import arch_model
            
            # Fit GARCH(1,1) model
            model = arch_model(returns.dropna() * 100, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Generate forecasts
            forecasts = fitted_model.forecast(horizon=horizon)
            
            return {
                'model_summary': fitted_model.summary(),
                'parameters': {
                    'omega': fitted_model.params['omega'],
                    'alpha[1]': fitted_model.params['alpha[1]'],
                    'beta[1]': fitted_model.params['beta[1]']
                },
                'conditional_volatility': fitted_model.conditional_volatility / 100,
                'forecast_variance': forecasts.variance.iloc[-1, :] / 10000,
                'forecast_volatility': np.sqrt(forecasts.variance.iloc[-1, :]) / 100,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        except ImportError:
            print("arch package not available for GARCH modeling")
            return self._simple_volatility_forecast(returns, horizon)
        except Exception as e:
            print(f"Error in GARCH modeling: {str(e)}")
            return self._simple_volatility_forecast(returns, horizon)
    
    def _simple_volatility_forecast(self, returns: pd.Series, horizon: int = 1) -> Dict:
        """Simple volatility forecast using EWMA"""
        lambda_ewma = 0.94  # RiskMetrics standard
        
        returns_clean = returns.dropna()
        
        # Calculate EWMA variance
        weights = np.array([(1 - lambda_ewma) * (lambda_ewma ** i) 
                           for i in range(len(returns_clean))])
        weights = weights[::-1] / weights.sum()
        
        ewma_variance = np.sum(weights * (returns_clean ** 2))
        ewma_volatility = np.sqrt(ewma_variance)
        
        # Simple forecast (assume persistence)
        forecast_volatility = ewma_volatility * np.sqrt(horizon)
        
        return {
            'parameters': {'lambda': lambda_ewma},
            'conditional_volatility': pd.Series([ewma_volatility] * len(returns_clean), 
                                               index=returns_clean.index),
            'forecast_volatility': np.array([forecast_volatility]),
            'forecast_variance': np.array([forecast_volatility ** 2])
        }
    
    def fama_french_risk_model(self, returns: pd.DataFrame, 
                              market_return: pd.Series,
                              risk_free_rate: pd.Series = None) -> Dict:
        """
        Fama-French three-factor risk model
        
        Args:
            returns: Asset returns DataFrame
            market_return: Market return series
            risk_free_rate: Risk-free rate series
            
        Returns:
            Dictionary with factor loadings and risk decomposition
        """
        if risk_free_rate is None:
            risk_free_rate = pd.Series(0.02/252, index=returns.index)  # 2% annual
        
        # Align all series
        aligned_data = pd.concat([returns, market_return, risk_free_rate], axis=1).dropna()
        
        if aligned_data.shape[1] < 3:
            raise ValueError("Insufficient data for Fama-French model")
        
        market_col = aligned_data.columns[-2]
        rf_col = aligned_data.columns[-1]
        
        # Calculate excess returns
        excess_market = aligned_data[market_col] - aligned_data[rf_col]
        
        results = {}
        
        for asset in returns.columns:
            if asset in aligned_data.columns:
                excess_asset = aligned_data[asset] - aligned_data[rf_col]
                
                # Simple CAPM regression (can be extended to 3-factor)
                X = excess_market.values.reshape(-1, 1)
                y = excess_asset.values
                
                # Add constant
                X_with_const = np.column_stack([np.ones(len(X)), X])
                
                # OLS regression
                try:
                    coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    alpha = coefficients[0]
                    beta = coefficients[1]
                    
                    # Calculate residuals and R-squared
                    predicted = X_with_const @ coefficients
                    residuals = y - predicted
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Systematic and idiosyncratic risk
                    systematic_var = (beta ** 2) * np.var(excess_market)
                    idiosyncratic_var = np.var(residuals)
                    total_var = np.var(excess_asset)
                    
                    results[asset] = {
                        'alpha': alpha * 252,  # Annualized
                        'beta': beta,
                        'r_squared': r_squared,
                        'systematic_risk': systematic_var * 252,
                        'idiosyncratic_risk': idiosyncratic_var * 252,
                        'total_risk': total_var * 252,
                        'tracking_error': np.sqrt(idiosyncratic_var) * np.sqrt(252)
                    }
                    
                except np.linalg.LinAlgError:
                    results[asset] = {
                        'alpha': 0, 'beta': 1, 'r_squared': 0,
                        'systematic_risk': 0, 'idiosyncratic_risk': 0,
                        'total_risk': 0, 'tracking_error': 0
                    }
        
        return results
    
    def extreme_value_theory_var(self, returns: pd.Series, 
                                confidence_level: float = 0.99,
                                threshold_percentile: float = 0.95) -> Dict:
        """
        Extreme Value Theory for tail risk estimation
        
        Args:
            returns: Return series
            confidence_level: VaR confidence level
            threshold_percentile: Threshold for extreme values
            
        Returns:
            Dictionary with EVT-based risk measures
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 100:
            return {'error': 'Insufficient data for EVT analysis'}
        
        # Define threshold for extreme values
        threshold = np.percentile(returns_clean, threshold_percentile * 100)
        
        # Extract excesses over threshold
        excesses = returns_clean[returns_clean > threshold] - threshold
        
        if len(excesses) < 10:
            return {'error': 'Insufficient extreme values'}
        
        # Fit Generalized Pareto Distribution (GPD)
        try:
            # Method of moments estimators for GPD
            mean_excess = np.mean(excesses)
            var_excess = np.var(excesses)
            
            # Shape parameter (xi)
            xi = 0.5 * (1 - (mean_excess ** 2) / var_excess)
            
            # Scale parameter (sigma)
            sigma = 0.5 * mean_excess * (1 + xi)
            
            # Number of exceedances
            n_excesses = len(excesses)
            n_total = len(returns_clean)
            
            # Probability of exceedance
            prob_exceed_threshold = n_excesses / n_total
            
            # EVT-based VaR
            if xi != 0:
                var_level = threshold + (sigma / xi) * (
                    ((n_total / n_excesses) * (1 - confidence_level)) ** (-xi) - 1
                )
            else:
                var_level = threshold - sigma * np.log(
                    (n_total / n_excesses) * (1 - confidence_level)
                )
            
            # Expected Shortfall (ES)
            if xi < 1 and xi != 0:
                es_level = var_level / (1 - xi) + (sigma - xi * threshold) / (1 - xi)
            else:
                es_level = var_level + sigma  # Approximation
            
            return {
                'threshold': threshold,
                'n_excesses': n_excesses,
                'xi_shape': xi,
                'sigma_scale': sigma,
                'var_evt': -var_level,  # Negative for loss
                'es_evt': -es_level,
                'tail_index': xi,
                'mean_excess_function': mean_excess
            }
            
        except Exception as e:
            print(f"Error in EVT analysis: {str(e)}")
            return {'error': f'EVT fitting failed: {str(e)}'}
    
    def dynamic_conditional_correlation(self, returns: pd.DataFrame, 
                                       window: int = 252) -> Dict:
        """
        Dynamic Conditional Correlation (DCC) model
        
        Args:
            returns: Multi-asset returns DataFrame
            window: Rolling window for correlation estimation
            
        Returns:
            Dictionary with dynamic correlations and analysis
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) < window:
            return {'error': 'Insufficient data for DCC model'}
        
        # Calculate rolling correlations
        rolling_corr = returns_clean.rolling(window=window).corr()
        
        # Extract correlation matrices for each date
        dates = rolling_corr.index.get_level_values(0).unique()[window-1:]
        assets = returns_clean.columns
        n_assets = len(assets)
        
        correlation_matrices = {}
        eigenvalues_over_time = []
        condition_numbers = []
        
        for date in dates:
            try:
                corr_matrix = rolling_corr.loc[date]
                
                # Ensure positive semi-definite
                eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
                eigenvals = np.maximum(eigenvals, 1e-8)  # Floor eigenvalues
                corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                # Normalize diagonal
                diag_sqrt = np.sqrt(np.diag(corr_matrix))
                corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
                
                correlation_matrices[date] = corr_matrix
                eigenvalues_over_time.append(eigenvals)
                condition_numbers.append(np.max(eigenvals) / np.min(eigenvals))
                
            except Exception as e:
                print(f"Error processing correlation matrix for {date}: {str(e)}")
                continue
        
        # Analyze correlation dynamics
        if correlation_matrices:
            # Average correlation
            avg_corr_matrix = np.mean(list(correlation_matrices.values()), axis=0)
            
            # Correlation stability (standard deviation of correlations)
            corr_time_series = np.array(list(correlation_matrices.values()))
            corr_volatility = np.std(corr_time_series, axis=0)
            
            # Off-diagonal correlation statistics
            mask = ~np.eye(n_assets, dtype=bool)
            avg_off_diag_corr = np.mean(avg_corr_matrix[mask])
            max_corr = np.max(avg_corr_matrix[mask])
            min_corr = np.min(avg_corr_matrix[mask])
            
            return {
                'correlation_matrices': correlation_matrices,
                'average_correlation_matrix': avg_corr_matrix,
                'correlation_volatility': corr_volatility,
                'average_correlation': avg_off_diag_corr,
                'max_correlation': max_corr,
                'min_correlation': min_corr,
                'eigenvalues_over_time': eigenvalues_over_time,
                'condition_numbers': condition_numbers,
                'avg_condition_number': np.mean(condition_numbers)
            }
        else:
            return {'error': 'Failed to compute dynamic correlations'}
    
    def risk_budgeting_portfolio(self, expected_returns: pd.Series,
                                covariance_matrix: pd.DataFrame,
                                risk_budgets: pd.Series) -> Dict:
        """
        Risk budgeting portfolio optimization
        
        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix
            risk_budgets: Target risk contribution for each asset
            
        Returns:
            Dictionary with optimal weights and risk contributions
        """
        n_assets = len(expected_returns)
        
        # Normalize risk budgets
        risk_budgets_norm = risk_budgets / risk_budgets.sum()
        
        def risk_budget_objective(weights):
            """Objective function for risk budgeting"""
            weights = np.array(weights)
            
            # Portfolio variance
            portfolio_var = weights.T @ covariance_matrix @ weights
            
            # Marginal risk contributions
            marginal_contrib = covariance_matrix @ weights
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib / portfolio_var
            
            # Objective: minimize squared differences from target risk budgets
            return np.sum((risk_contrib - risk_budgets_norm) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (non-negative weights)
        bounds = [(0.001, 0.999) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                risk_budget_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate final risk contributions
                portfolio_var = optimal_weights.T @ covariance_matrix @ optimal_weights
                marginal_contrib = covariance_matrix @ optimal_weights
                risk_contrib = optimal_weights * marginal_contrib / portfolio_var
                
                # Expected portfolio return
                portfolio_return = optimal_weights.T @ expected_returns
                portfolio_vol = np.sqrt(portfolio_var)
                
                return {
                    'optimal_weights': pd.Series(optimal_weights, index=expected_returns.index),
                    'risk_contributions': pd.Series(risk_contrib, index=expected_returns.index),
                    'target_risk_budgets': risk_budgets_norm,
                    'portfolio_return': portfolio_return,
                    'portfolio_volatility': portfolio_vol,
                    'optimization_success': True,
                    'objective_value': result.fun
                }
            else:
                return {'optimization_success': False, 'error': result.message}
                
        except Exception as e:
            return {'optimization_success': False, 'error': str(e)}
    
    def black_litterman_model(self, market_caps: pd.Series,
                             covariance_matrix: pd.DataFrame,
                             risk_aversion: float = 3.0,
                             views: Dict = None,
                             view_uncertainty: float = 0.1) -> Dict:
        """
        Black-Litterman model for expected returns
        
        Args:
            market_caps: Market capitalizations for assets
            covariance_matrix: Historical covariance matrix
            risk_aversion: Risk aversion parameter
            views: Dictionary of investor views {asset: expected_return}
            view_uncertainty: Uncertainty in views (higher = less confident)
            
        Returns:
            Dictionary with Black-Litterman expected returns and optimal weights
        """
        # Market capitalization weights
        market_weights = market_caps / market_caps.sum()
        
        # Implied equilibrium returns
        pi = risk_aversion * covariance_matrix @ market_weights
        
        if views is None or len(views) == 0:
            # No views - return market portfolio
            return {
                'implied_returns': pi,
                'bl_returns': pi,
                'optimal_weights': market_weights,
                'market_weights': market_weights
            }
        
        # Convert views to arrays
        assets = list(market_weights.index)
        view_assets = [asset for asset in views.keys() if asset in assets]
        
        if not view_assets:
            return {
                'implied_returns': pi,
                'bl_returns': pi,
                'optimal_weights': market_weights,
                'market_weights': market_weights,
                'error': 'No valid views found'
            }
        
        # Create picking matrix P and view vector Q
        P = np.zeros((len(view_assets), len(assets)))
        Q = np.zeros(len(view_assets))
        
        for i, asset in enumerate(view_assets):
            asset_idx = assets.index(asset)
            P[i, asset_idx] = 1.0
            Q[i] = views[asset]
        
        # View uncertainty matrix (diagonal)
        omega = np.eye(len(view_assets)) * view_uncertainty
        
        # Tau parameter (scales the uncertainty of the prior)
        tau = 1.0 / len(market_weights)
        
        # Black-Litterman formula
        try:
            # Precision matrices
            M1 = np.linalg.inv(tau * covariance_matrix)
            M2 = P.T @ np.linalg.inv(omega) @ P
            M3 = np.linalg.inv(tau * covariance_matrix) @ pi
            M4 = P.T @ np.linalg.inv(omega) @ Q
            
            # New expected returns
            bl_returns = np.linalg.inv(M1 + M2) @ (M3 + M4)
            
            # New covariance matrix
            bl_covariance = np.linalg.inv(M1 + M2)
            
            # Optimal weights
            optimal_weights = (1 / risk_aversion) * np.linalg.inv(bl_covariance) @ bl_returns
            
            return {
                'implied_returns': pi,
                'bl_returns': pd.Series(bl_returns, index=assets),
                'bl_covariance': pd.DataFrame(bl_covariance, index=assets, columns=assets),
                'optimal_weights': pd.Series(optimal_weights, index=assets),
                'market_weights': market_weights,
                'views': views,
                'tau': tau
            }
            
        except np.linalg.LinAlgError as e:
            return {
                'implied_returns': pi,
                'bl_returns': pi,
                'optimal_weights': market_weights,
                'market_weights': market_weights,
                'error': f'Matrix inversion failed: {str(e)}'
            }
    
    def portfolio_risk_attribution(self, weights: pd.Series,
                                  covariance_matrix: pd.DataFrame,
                                  factor_loadings: pd.DataFrame = None) -> Dict:
        """
        Portfolio risk attribution analysis
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            factor_loadings: Factor loadings matrix (optional)
            
        Returns:
            Dictionary with risk attribution results
        """
        # Portfolio variance
        portfolio_var = weights.T @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal contributions to risk
        marginal_contrib = covariance_matrix @ weights
        
        # Component contributions to risk
        component_contrib = weights * marginal_contrib
        
        # Percentage contributions
        pct_contrib = component_contrib / portfolio_var
        
        # Risk decomposition
        attribution = {
            'portfolio_variance': portfolio_var,
            'portfolio_volatility': portfolio_vol,
            'marginal_contributions': pd.Series(marginal_contrib, index=weights.index),
            'component_contributions': pd.Series(component_contrib, index=weights.index),
            'percentage_contributions': pd.Series(pct_contrib, index=weights.index)
        }
        
        # Factor-based attribution if factor loadings provided
        if factor_loadings is not None:
            try:
                # Factor portfolio exposures
                factor_exposures = factor_loadings.T @ weights
                
                # Factor covariance matrix
                factor_returns = factor_loadings.T @ covariance_matrix @ factor_loadings
                
                # Factor contributions to portfolio risk
                factor_contrib = factor_exposures.T @ factor_returns @ factor_exposures
                
                attribution['factor_exposures'] = factor_exposures
                attribution['factor_contributions'] = factor_contrib
                
            except Exception as e:
                attribution['factor_error'] = str(e)
        
        return attribution

# Example usage
if __name__ == "__main__":
    # Sample usage with synthetic data
    np.random.seed(42)
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_assets = 4
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    # Simulate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.6, 0.7, 0.4],
        [0.6, 1.0, 0.5, 0.3],
        [0.7, 0.5, 1.0, 0.4],
        [0.4, 0.3, 0.4, 1.0]
    ])
    
    volatilities = np.array([0.025, 0.030, 0.022, 0.045])  # Daily volatilities
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(
        mean=[0.0008, 0.0006, 0.0007, 0.0005],  # Daily expected returns
        cov=covariance_matrix,
        size=len(dates)
    )
    
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    # Initialize risk models
    risk_models = RiskModels()
    
    print("=== ADVANCED RISK MODELS DEMONSTRATION ===\n")
    
    # 1. GARCH volatility forecasting
    print("1. GARCH VOLATILITY FORECAST (AAPL)")
    garch_results = risk_models.garch_volatility_forecast(returns_df['AAPL'], horizon=5)
    if 'parameters' in garch_results:
        print(f"Current volatility: {garch_results['conditional_volatility'].iloc[-1]:.4f}")
        print(f"5-day forecast volatility: {garch_results['forecast_volatility'][0]:.4f}")
    print()
    
    # 2. Fama-French risk model
    print("2. FAMA-FRENCH RISK MODEL")
    market_return = returns_df.mean(axis=1)  # Equal-weighted market proxy
    ff_results = risk_models.fama_french_risk_model(returns_df, market_return)
    
    for asset, metrics in ff_results.items():
        print(f"{asset}: Beta={metrics['beta']:.3f}, Alpha={metrics['alpha']:.4f}, R²={metrics['r_squared']:.3f}")
    print()
    
    # 3. Extreme Value Theory
    print("3. EXTREME VALUE THEORY (TSLA)")
    evt_results = risk_models.extreme_value_theory_var(returns_df['TSLA'])
    if 'var_evt' in evt_results:
        print(f"EVT VaR (99%): {evt_results['var_evt']:.4f}")
        print(f"EVT Expected Shortfall: {evt_results['es_evt']:.4f}")
        print(f"Tail Index (ξ): {evt_results['xi_shape']:.4f}")
    print()
    
    # 4. Dynamic Conditional Correlation
    print("4. DYNAMIC CONDITIONAL CORRELATION")
    dcc_results = risk_models.dynamic_conditional_correlation(returns_df, window=100)
    if 'average_correlation' in dcc_results:
        print(f"Average Correlation: {dcc_results['average_correlation']:.3f}")
        print(f"Max Correlation: {dcc_results['max_correlation']:.3f}")
        print(f"Min Correlation: {dcc_results['min_correlation']:.3f}")
        print(f"Average Condition Number: {dcc_results['avg_condition_number']:.2f}")
    print()
    
    # 5. Risk Budgeting Portfolio
    print("5. RISK BUDGETING PORTFOLIO")
    expected_returns = returns_df.mean() * 252  # Annualized
    annual_cov = returns_df.cov() * 252
    risk_budgets = pd.Series([0.25, 0.25, 0.25, 0.25], index=asset_names)  # Equal risk
    
    rb_results = risk_models.risk_budgeting_portfolio(expected_returns, annual_cov, risk_budgets)
    if rb_results['optimization_success']:
        print("Optimal Weights:")
        for asset, weight in rb_results['optimal_weights'].items():
            print(f"  {asset}: {weight:.3f}")
        print(f"Portfolio Return: {rb_results['portfolio_return']:.4f}")
        print(f"Portfolio Volatility: {rb_results['portfolio_volatility']:.4f}")
    print()
    
    # 6. Black-Litterman Model
    print("6. BLACK-LITTERMAN MODEL")
    market_caps = pd.Series([2000, 1500, 1800, 800], index=asset_names)  # Mock market caps
    views = {'AAPL': 0.12, 'TSLA': 0.15}  # Bullish views
    
    bl_results = risk_models.black_litterman_model(market_caps, annual_cov, views=views)
    if 'bl_returns' in bl_results:
        print("Black-Litterman Expected Returns:")
        for asset, ret in bl_results['bl_returns'].items():
            print(f"  {asset}: {ret:.4f}")
        print("\nOptimal Weights:")
        for asset, weight in bl_results['optimal_weights'].items():
            print(f"  {asset}: {weight:.3f}")
    print()
    
    # 7. Portfolio Risk Attribution
    print("7. PORTFOLIO RISK ATTRIBUTION")
    portfolio_weights = pd.Series([0.3, 0.25, 0.25, 0.2], index=asset_names)
    attribution = risk_models.portfolio_risk_attribution(portfolio_weights, annual_cov)
    
    print(f"Portfolio Volatility: {attribution['portfolio_volatility']:.4f}")
    print("Risk Contributions:")
    for asset, contrib in attribution['percentage_contributions'].items():
        print(f"  {asset}: {contrib:.1%}")
    
    print("\n=== RISK MODELS ANALYSIS COMPLETE ===")