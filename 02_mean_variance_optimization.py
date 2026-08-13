"""
Portfolio Optimization Engine - Day 2: Mean-Variance Optimization
Implements mean-variance optimization using cvxpy with basic constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize

# Constants for the optimization
RISK_AVERSION = 2.5  # Risk aversion coefficient (λ)

def mean_variance_optimization(returns_df: pd.DataFrame, 
                               expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame,
                               target_return: float = None,
                               risk_aversion: float = RISK_AVERSION) -> dict:
    """
    Solve the mean-variance optimization problem using cvxpy.
    
    The optimization problem is:
    Maximize: w^T μ - (λ/2) w^T Σ w
    Subject to: sum(w) = 1
                w >= 0 (long-only)
                w^T μ >= target_return (if provided)
    
    Args:
        returns_df: DataFrame of historical returns
        expected_returns: Series of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        target_return: Desired target return (if None, maximize utility)
        risk_aversion: Coefficient of risk aversion
    
    Returns:
        Dictionary containing optimal weights and portfolio statistics
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)
    
    # Objective: Maximize w^T μ - (λ/2) w^T Σ w
    objective = cp.Maximize(
        expected_returns.values @ weights - 
        (risk_aversion / 2) * cp.quad_form(weights, cov_matrix.values)
    )
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1,          # Full investment
        weights >= 0                   # Long-only
    ]
    
    # Add target return constraint if specified
    if target_return is not None:
        constraints.append(expected_returns.values @ weights >= target_return)
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    # Check if optimization was successful
    if problem.status != cp.OPTIMAL:
        print(f"Warning: Optimization status: {problem.status}")
        return None
    
    # Get optimal weights
    optimal_weights = pd.Series(weights.value, index=expected_returns.index)
    
    # Calculate portfolio statistics
    portfolio_return = (expected_returns * optimal_weights).sum()
    portfolio_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
        'status': problem.status,
        'objective_value': problem.value
    }

def maximize_sharpe_ratio(returns_df: pd.DataFrame, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """
    Maximize the Sharpe ratio of the portfolio.
    
    Args:
        returns_df: DataFrame of historical returns
        expected_returns: Series of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    
    Returns:
        Dictionary containing optimal weights and portfolio statistics
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)
    
    # Portfolio return and risk
    portfolio_return = expected_returns.values @ weights
    portfolio_risk = cp.sqrt(cp.quad_form(weights, cov_matrix.values))
    
    # Objective: Maximize Sharpe ratio = (w^T μ - r_f) / sqrt(w^T Σ w)
    # cvxpy doesn't directly support ratio objectives, so we use a reformulation
    # Maximize w^T μ subject to sqrt(w^T Σ w) = 1 and w >= 0, sum(w) = 1
    # This is equivalent to maximizing the Sharpe ratio
    
    # Instead, we can find the maximum Sharpe ratio by solving a series of optimization problems
    # or using the fact that the maximum Sharpe ratio portfolio lies on the efficient frontier
    
    # Use scipy to find the maximum Sharpe ratio
    def neg_sharpe(weights):
        # Negative Sharpe ratio for minimization
        portfolio_return = np.sum(expected_returns.values * weights)
        portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
        if portfolio_risk == 0:
            return 1e10
        return -(portfolio_return - risk_free_rate) / portfolio_risk
    
    # Constraints: sum(weights) = 1, weights >= 0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # Initial guess: equal weights
    init_weights = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(neg_sharpe, init_weights, method='SLSQP', 
                     constraints=constraints, bounds=bounds)
    
    if not result.success:
        print(f"Warning: Optimization failed: {result.message}")
        return None
    
    optimal_weights = pd.Series(result.x, index=expected_returns.index)
    
    # Calculate portfolio statistics
    portfolio_return = (expected_returns * optimal_weights).sum()
    portfolio_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe': sharpe,
        'status': 'Optimal',
        'objective_value': -result.fun
    }

def solve_optimal_portfolio(returns_df: pd.DataFrame, expected_returns: pd.Series, cov_matrix: pd.DataFrame, target_returns: list = None, risk_free_rate: float = 0.02) -> dict:
    """
    Solve for the optimal portfolio weights.
    
    Args:
        returns_df: DataFrame of historical returns
        expected_returns: Series of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        target_returns: List of target returns for efficient frontier
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    
    Returns:
        Dictionary containing optimal portfolios and statistics
    """
    # If no target returns provided, create a range
    if target_returns is None:
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, 20)
    
    # Optimize for each target return
    optimal_portfolios = []
    for target in target_returns:
        result = mean_variance_optimization(
            returns_df, expected_returns, cov_matrix, 
            target_return=target, risk_aversion=RISK_AVERSION
        )
        if result is not None:
            optimal_portfolios.append(result)
    
    # Get maximum Sharpe ratio portfolio
    max_sharpe_portfolio = maximize_sharpe_ratio(
        returns_df, expected_returns, cov_matrix, risk_free_rate
    )
    
    return {
        'frontier': optimal_portfolios,
        'max_sharpe': max_sharpe_portfolio
    }

def print_optimization_results(results: dict) -> None:
    """
    Print the optimization results in a readable format.
    
    Args:
        results: Dictionary containing optimization results
    """
    print("=" * 70)
    print("MEAN-VARIANCE OPTIMIZATION RESULTS")
    print("=" * 70)
    
    # Print maximum Sharpe portfolio
    if results['max_sharpe'] is not None:
        max_sharpe = results['max_sharpe']
        print("\nMAXIMUM SHARPE PORTFOLIO")
        print("-" * 70)
        print(f"Portfolio Return: {max_sharpe['return']:.4f} ({max_sharpe['return']*100:.2f}%)")
        print(f"Portfolio Risk:    {max_sharpe['risk']:.4f} ({max_sharpe['risk']*100:.2f}%)")
        print(f"Sharpe Ratio:      {max_sharpe['sharpe']:.4f}")
        print("\nWeights (Top 10):")
        top_weights = max_sharpe['weights'].sort_values(ascending=False).head(10)
        for asset, weight in top_weights.items():
            print(f"  {asset}: {weight*100:.2f}%")
    
    # Print efficient frontier summary
    if results['frontier']:
        print("\n" + "=" * 70)
        print("EFFICIENT FRONTIER SUMMARY")
        print("=" * 70)
        print(f"{'Return':<12} {'Risk':<12} {'Sharpe':<12}")
        print("-" * 70)
        for portfolio in results['frontier'][::2]:  # Print every other point
            print(f"{portfolio['return']:<12.4f} {portfolio['risk']:<12.4f} {portfolio['sharpe']:<12.4f}")

def plot_efficient_frontier(results: dict, risk_free_rate: float = 0.02) -> None:
    """
    Plot the efficient frontier and optimal portfolios.
    
    Args:
        results: Dictionary containing optimization results
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    """
    plt.figure(figsize=(12, 8))
    
    # Plot efficient frontier
    if results['frontier']:
        returns = [p['return'] for p in results['frontier']]
        risks = [p['risk'] for p in results['frontier']]
        plt.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot maximum Sharpe portfolio
    if results['max_sharpe'] is not None:
        max_sharpe = results['max_sharpe']
        plt.scatter(max_sharpe['risk'], max_sharpe['return'], 
                   color='red', s=100, zorder=5, label='Max Sharpe')
    
    # Plot individual assets
    # This would require asset returns and risks which are not provided directly
    
    # Plot capital market line
    if results['max_sharpe'] is not None:
        max_sharpe = results['max_sharpe']
        x_cml = np.array([0, max_sharpe['risk'] * 1.5])
        y_cml = risk_free_rate + (max_sharpe['return'] - risk_free_rate) / max_sharpe['risk'] * x_cml
        plt.plot(x_cml, y_cml, 'g--', linewidth=1.5, label='Capital Market Line')
    
    plt.xlabel('Portfolio Risk (σ)')
    plt.ylabel('Expected Return (μ)')
    plt.title('Efficient Frontier and Optimal Portfolios')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Run the mean-variance optimization demonstration.
    """
    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION ENGINE - DAY 2: MEAN-VARIANCE OPTIMIZATION")
    print("=" * 70)
    
    # Load data from Day 1
    # Note: In practice, this would load from CSV files
    # For this demonstration, we'll assume data is already generated
    
    print("Loading data from Day 1...")
    
    # For demonstration, let's assume data is generated from Day 1
    # This is a placeholder for the actual data loading
    # In practice, you would:
    # returns_df = pd.read_csv('data/asset_returns.csv', index_col=0, parse_dates=True)
    # expected_returns = pd.read_csv('data/expected_returns.csv', index_col=0).squeeze()
    # cov_matrix = pd.read_csv('data/covariance_matrix.csv', index_col=0)
    
    # Generate synthetic data directly
    np.random.seed(42)
    n_assets = 30
    n_days = 252 * 3
    
    # Generate returns
    volatilities = np.random.uniform(0.10, 0.40, n_assets)
    correlations = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
    correlations = (correlations + correlations.T) / 2
    np.fill_diagonal(correlations, 1.0)
    
    # Ensure positive definite
    eigenvals, eigenvecs = np.linalg.eigh(correlations)
    eigenvals = np.maximum(eigenvals, 1e-6)
    correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Generate returns
    daily_vols = volatilities / np.sqrt(252)
    cholesky = np.linalg.cholesky(correlations)
    returns = np.random.normal(0, 1, (n_days, n_assets))
    returns = returns @ cholesky.T * daily_vols
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
    asset_names = [f'Asset_{i+1:02d}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    # Calculate expected returns and covariance
    expected_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    print(f"Number of Assets: {len(returns_df.columns)}")
    print(f"Date Range: {returns_df.index[0]} to {returns_df.index[-1]}")
    print("\n")
    
    # Optimize portfolio
    print("Optimizing portfolio...")
    results = solve_optimal_portfolio(returns_df, expected_returns, cov_matrix)
    
    # Print results
    print_optimization_results(results)
    
    # Plot efficient frontier
    plot_efficient_frontier(results, risk_free_rate=0.02)
    
    print("\n" + "=" * 70)
    print("Optimization complete. Ready for Day 3: Efficient Frontier Analysis.")
    print("=" * 70)

if __name__ == "__main__":
    main()
