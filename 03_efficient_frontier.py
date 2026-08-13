"""
Portfolio Optimization Engine - Day 3: Efficient Frontier
Computes and analyzes the efficient frontier for the portfolio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize

def calculate_efficient_frontier(returns_df: pd.DataFrame,
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  n_points: int = 50,
                                  risk_free_rate: float = 0.02) -> dict:
    """
    Calculate the efficient frontier for the given assets.
    
    Args:
        returns_df: DataFrame of historical returns
        expected_returns: Series of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        n_points: Number of points on the efficient frontier
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    
    Returns:
        Dictionary containing frontier portfolios and statistics
    """
    n_assets = len(expected_returns)
    
    # Find minimum and maximum possible returns
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    
    # Generate target returns for efficient frontier
    target_returns = np.linspace(min_return, max_return, n_points)
    
    frontier_portfolios = []
    
    for target in target_returns:
        # Solve for minimum variance at target return
        weights = cp.Variable(n_assets)
        
        objective = cp.Minimize(cp.quad_form(weights, cov_matrix.values))
        constraints = [
            cp.sum(weights) == 1,      # Full investment
            weights >= 0,               # Long-only
            expected_returns.values @ weights >= target  # Target return
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status == cp.OPTIMAL:
            optimal_weights = pd.Series(weights.value, index=expected_returns.index)
            portfolio_return = (expected_returns * optimal_weights).sum()
            portfolio_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            frontier_portfolios.append({
                'weights': optimal_weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe': sharpe,
                'target_return': target
            })
    
    return {
        'frontier': frontier_portfolios,
        'min_return': min_return,
        'max_return': max_return
    }

def find_minimum_variance_portfolio(returns_df: pd.DataFrame,
                                    expected_returns: pd.Series,
                                    cov_matrix: pd.DataFrame) -> dict:
    """
    Find the minimum variance portfolio (MVP).
    
    Args:
        returns_df: DataFrame of historical returns
        expected_returns: Series of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
    
    Returns:
        Dictionary containing MVP weights and statistics
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)
    
    # Minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix.values))
    constraints = [
        cp.sum(weights) == 1,  # Full investment
        weights >= 0           # Long-only
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    if problem.status != cp.OPTIMAL:
        print(f"Warning: MVP optimization status: {problem.status}")
        return None
    
    optimal_weights = pd.Series(weights.value, index=expected_returns.index)
    portfolio_return = (expected_returns * optimal_weights).sum()
    portfolio_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
        'status': problem.status
    }

def find_tangency_portfolio(returns_df: pd.DataFrame,
                            expected_returns: pd.Series,
                            cov_matrix: pd.DataFrame,
                            risk_free_rate: float = 0.02) -> dict:
    """
    Find the tangency portfolio (maximum Sharpe ratio).
    
    Args:
        returns_df: DataFrame of historical returns
        expected_returns: Series of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    
    Returns:
        Dictionary containing tangency portfolio weights and statistics
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)
    
    # Maximize Sharpe ratio: (w^T μ - r_f) / sqrt(w^T Σ w)
    # This is equivalent to maximizing w^T μ subject to sqrt(w^T Σ w) = 1
    # But cvxpy doesn't handle ratio objectives directly
    
    # Use scipy optimization for the tangency portfolio
    def neg_sharpe(weights):
        portfolio_return = np.sum(expected_returns.values * weights)
        portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
        if portfolio_risk == 0:
            return 1e10
        return -(portfolio_return - risk_free_rate) / portfolio_risk
    
    # Constraints: sum(weights) = 1, weights >= 0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_assets)]
    init_weights = np.ones(n_assets) / n_assets
    
    result = minimize(neg_sharpe, init_weights, method='SLSQP',
                     constraints=constraints, bounds=bounds)
    
    if not result.success:
        print(f"Warning: Tangency portfolio optimization failed: {result.message}")
        return None
    
    optimal_weights = pd.Series(result.x, index=expected_returns.index)
    portfolio_return = (expected_returns * optimal_weights).sum()
    portfolio_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe': sharpe,
        'status': 'Optimal'
    }

def plot_efficient_frontier_with_assets(frontier_data: dict,
                                         returns_df: pd.DataFrame,
                                         expected_returns: pd.Series,
                                         cov_matrix: pd.DataFrame,
                                         mvp: dict,
                                         tangency: dict,
                                         risk_free_rate: float = 0.02) -> None:
    """
    Plot the efficient frontier with individual assets and optimal portfolios.
    
    Args:
        frontier_data: Dictionary containing efficient frontier data
        returns_df: DataFrame of historical returns
        expected_returns: Series of expected returns
        cov_matrix: Covariance matrix
        mvp: Minimum variance portfolio results
        tangency: Tangency portfolio results
        risk_free_rate: Risk-free rate
    """
    plt.figure(figsize=(14, 10))
    
    # Plot efficient frontier
    if frontier_data['frontier']:
        frontier_returns = [p['return'] for p in frontier_data['frontier']]
        frontier_risks = [p['risk'] for p in frontier_data['frontier']]
        plt.plot(frontier_risks, frontier_returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot individual assets
    asset_returns = [expected_returns[asset] for asset in returns_df.columns]
    asset_risks = [np.sqrt(cov_matrix.loc[asset, asset]) for asset in returns_df.columns]
    plt.scatter(asset_risks, asset_returns, color='gray', alpha=0.6, label='Individual Assets')
    
    # Annotate assets (only a subset to avoid clutter)
    for i, asset in enumerate(returns_df.columns):
        if i % 5 == 0:  # Annotate every 5th asset
            plt.annotate(asset, (asset_risks[i], asset_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot MVP
    if mvp is not None:
        plt.scatter(mvp['risk'], mvp['return'], color='green', s=150, 
                   marker='s', zorder=5, label='Minimum Variance Portfolio')
        plt.annotate('MVP', (mvp['risk'], mvp['return']), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    # Plot tangency portfolio
    if tangency is not None:
        plt.scatter(tangency['risk'], tangency['return'], color='red', s=150,
                   marker='D', zorder=5, label='Tangency Portfolio (Max Sharpe)')
        plt.annotate('Tangency', (tangency['risk'], tangency['return']),
                    xytext=(10, -20), textcoords='offset points', fontsize=10)
    
    # Plot capital market line
    if tangency is not None:
        x_cml = np.array([0, tangency['risk'] * 1.5])
        y_cml = risk_free_rate + (tangency['return'] - risk_free_rate) / tangency['risk'] * x_cml
        plt.plot(x_cml, y_cml, 'g--', linewidth=1.5, label='Capital Market Line')
        plt.scatter(0, risk_free_rate, color='orange', s=100, zorder=5, label='Risk-Free Rate')
    
    plt.xlabel('Portfolio Risk (σ)')
    plt.ylabel('Expected Return (μ)')
    plt.title('Efficient Frontier with Optimal Portfolios')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_frontier_characteristics(frontier_data: dict) -> None:
    """
    Analyze and print characteristics of the efficient frontier.
    
    Args:
        frontier_data: Dictionary containing efficient frontier data
    """
    print("=" * 70)
    print("EFFICIENT FRONTIER ANALYSIS")
    print("=" * 70)
    
    if not frontier_data['frontier']:
        print("No frontier data available.")
        return
    
    # Calculate curvature of the frontier
    frontier_returns = [p['return'] for p in frontier_data['frontier']]
    frontier_risks = [p['risk'] for p in frontier_data['frontier']]
    
    # Find the point of maximum curvature
    curvature = []
    for i in range(1, len(frontier_risks) - 1):
        dx1 = frontier_risks[i] - frontier_risks[i-1]
        dx2 = frontier_risks[i+1] - frontier_risks[i]
        dy1 = frontier_returns[i] - frontier_returns[i-1]
        dy2 = frontier_returns[i+1] - frontier_returns[i]
        
        if dx1 > 0 and dx2 > 0:
            # Approximate curvature
            slope1 = dy1 / dx1
            slope2 = dy2 / dx2
            curvature.append(abs(slope2 - slope1))
        else:
            curvature.append(0)
    
    if curvature:
        max_curvature_idx = np.argmax(curvature) + 1
        print(f"\nPoint of maximum curvature at return: {frontier_returns[max_curvature_idx]:.4f}")
        print(f"Corresponding risk: {frontier_risks[max_curvature_idx]:.4f}")
        
        # Calculate efficiency of the frontier
        # Compare to a straight line from min to max
        min_return = frontier_returns[0]
        max_return = frontier_returns[-1]
        min_risk = frontier_risks[0]
        max_risk = frontier_risks[-1]
        
        # Area under the frontier (approximated)
        area_under_frontier = np.trapz(frontier_returns, frontier_risks)
        area_under_line = np.trapz([min_return, max_return], [min_risk, max_risk])
        
        efficiency_ratio = area_under_frontier / area_under_line if area_under_line > 0 else 0
        print(f"\nEfficiency ratio (area under frontier / area under line): {efficiency_ratio:.3f}")
        print(f"(Higher values indicate a more convex frontier)")

def print_portfolio_weights(portfolio: dict, title: str = "Portfolio") -> None:
    """
    Print the weights of a portfolio in a readable format.
    
    Args:
        portfolio: Dictionary containing portfolio weights and statistics
        title: Title for the portfolio
    """
    if portfolio is None:
        print(f"{title}: Not available")
        return
    
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    print(f"Return: {portfolio['return']:.4f} ({portfolio['return']*100:.2f}%)")
    print(f"Risk:   {portfolio['risk']:.4f} ({portfolio['risk']*100:.2f}%)")
    print(f"Sharpe: {portfolio['sharpe']:.4f}")
    print("\nTop 10 Weights:")
    weights = portfolio['weights'].sort_values(ascending=False).head(10)
    for asset, weight in weights.items():
        print(f"  {asset}: {weight*100:.2f}%")

def main():
    """
    Run efficient frontier analysis.
    """
    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION ENGINE - DAY 3: EFFICIENT FRONTIER")
    print("=" * 70)
    
    # Generate synthetic data (same as Day 2)
    np.random.seed(42)
    n_assets = 30
    n_days = 252 * 3
    
    # Generate returns
    volatilities = np.random.uniform(0.10, 0.40, n_assets)
    correlations = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
    correlations = (correlations + correlations.T) / 2
    np.fill_diagonal(correlations, 1.0)
    
    eigenvals, eigenvecs = np.linalg.eigh(correlations)
    eigenvals = np.maximum(eigenvals, 1e-6)
    correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
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
    
    print(f"\nAssets: {len(returns_df.columns)}")
    print(f"Date Range: {returns_df.index[0]} to {returns_df.index[-1]}")
    
    # Calculate efficient frontier
    print("\nCalculating efficient frontier...")
    frontier_data = calculate_efficient_frontier(returns_df, expected_returns, cov_matrix)
    
    # Find MVP
    print("\nFinding minimum variance portfolio...")
    mvp = find_minimum_variance_portfolio(returns_df, expected_returns, cov_matrix)
    
    # Find tangency portfolio
    print("\nFinding tangency portfolio...")
    tangency = find_tangency_portfolio(returns_df, expected_returns, cov_matrix)
    
    # Print portfolio weights
    print_portfolio_weights(mvp, "MINIMUM VARIANCE PORTFOLIO")
    print_portfolio_weights(tangency, "TANGENCY PORTFOLIO (MAX SHARPE)")
    
    # Analyze frontier
    analyze_frontier_characteristics(frontier_data)
    
    # Plot
    plot_efficient_frontier_with_assets(
        frontier_data, returns_df, expected_returns, cov_matrix, 
        mvp, tangency
    )
    
    print("\n" + "=" * 70)
    print("Efficient frontier analysis complete.")
    print("Ready for Day 4: Black-Litterman Integration.")
    print("=" * 70)

if __name__ == "__main__":
    main()
