"""
Portfolio Optimization Engine - Day 1: Data Preparation
Loads asset return data, calculates expected returns, and estimates the covariance matrix.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Configuration parameters
N_ASSETS = 35  # Number of assets in the portfolio
N_DAYS = 252   # Trading days per year
LOOKBACK_YEARS = 3  # Historical data period

def generate_synthetic_returns(n_assets: int = N_ASSETS, 
                              n_days: int = N_DAYS * LOOKBACK_YEARS,
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic asset returns with realistic correlations and volatilities.
    
    Args:
        n_assets: Number of assets to generate
        n_days: Number of trading days of history
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame of daily returns with assets as columns and dates as index
    """
    np.random.seed(seed)
    
    # Generate realistic asset volatilities (10% to 40% annualized)
    volatilities = np.random.uniform(0.10, 0.40, n_assets)
    
    # Generate correlations (0.2 to 0.8, with some variation)
    # Use random positive definite correlation matrix
    corr = np.random.uniform(0.2, 0.8, (n_assets, n_assets))
    corr = (corr + corr.T) / 2  # Symmetrize
    np.fill_diagonal(corr, 1.0)  # Set diagonal to 1
    
    # Convert to positive definite if needed
    eigenvals, eigenvecs = np.linalg.eigh(corr)
    eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive eigenvalues
    corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Generate daily returns with specified volatilities and correlation
    daily_vols = volatilities / np.sqrt(252)  # Convert to daily volatility
    cholesky = np.linalg.cholesky(corr)
    
    # Generate correlated normal returns
    daily_returns = np.random.normal(0, 1, (n_days, n_assets))
    daily_returns = daily_returns @ cholesky.T * daily_vols
    
    # Convert to DataFrame
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    asset_names = [f'Asset_{i+1:02d}' for i in range(n_assets)]
    
    returns_df = pd.DataFrame(daily_returns, index=dates, columns=asset_names)
    
    # Add sector classification for later use
    sectors = ['Technology', 'Financials', 'Healthcare', 'Consumer', 'Energy']
    sector_map = {asset: np.random.choice(sectors) for asset in asset_names}
    
    # Store sector info
    returns_df.attrs['sector_map'] = sector_map
    
    return returns_df

def calculate_expected_returns(returns_df: pd.DataFrame, method: str = 'historical') -> pd.Series:
    """
    Calculate expected returns for each asset.
    
    Args:
        returns_df: DataFrame of historical returns
        method: Method to use ('historical', 'ewm', 'shrinkage')
    
    Returns:
        Series of expected annual returns
    """
    if method == 'historical':
        # Simple historical average annualized
        expected_returns = returns_df.mean() * 252
    elif method == 'ewm':
        # Exponentially weighted moving average (more weight to recent data)
        expected_returns = returns_df.ewm(span=60).mean().iloc[-1] * 252
    elif method == 'shrinkage':
        # Shrinkage estimator toward mean across assets
        historical = returns_df.mean() * 252
        grand_mean = historical.mean()
        lambda_shrink = 0.5  # Shrinkage intensity
        expected_returns = (1 - lambda_shrink) * historical + lambda_shrink * grand_mean
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return expected_returns

def calculate_covariance_matrix(returns_df: pd.DataFrame, method: str = 'sample') -> pd.DataFrame:
    """
    Estimate covariance matrix for all assets.
    
    Args:
        returns_df: DataFrame of historical returns
        method: Method to use ('sample', 'ledoit-wolf', 'ewm')
    
    Returns:
        Covariance matrix as DataFrame
    """
    if method == 'sample':
        # Simple sample covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualize covariance
    elif method == 'ledoit-wolf':
        # Ledoit-Wolf shrinkage estimator
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns_df)
        cov_matrix = pd.DataFrame(lw.covariance_ * 252, 
                                   index=returns_df.columns, 
                                   columns=returns_df.columns)
    elif method == 'ewm':
        # Exponentially weighted moving average covariance
        ewm_cov = returns_df.ewm(span=60).cov().iloc[-len(returns_df.columns):]
        cov_matrix = ewm_cov * 252
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return cov_matrix

def calculate_implied_equilibrium_returns(cov_matrix: pd.DataFrame, 
                                         market_weights: pd.Series,
                                         risk_aversion: float = 2.5) -> pd.Series:
    """
    Calculate implied equilibrium returns (Black-Litterman reverse optimization).
    
    Args:
        cov_matrix: Covariance matrix of asset returns
        market_weights: Market capitalization weights (must sum to 1)
        risk_aversion: Coefficient of risk aversion
    
    Returns:
        Implied equilibrium returns
    """
    # Ensure market weights sum to 1
    market_weights = market_weights / market_weights.sum()
    
    # Implied returns: Π = λ * Σ * w_mkt
    implied_returns = risk_aversion * cov_matrix @ market_weights
    
    return implied_returns

def generate_market_weights(n_assets: int, seed: int = 42) -> pd.Series:
    """
    Generate synthetic market capitalization weights.
    
    Args:
        n_assets: Number of assets
        seed: Random seed for reproducibility
    
    Returns:
        Series of market weights (sums to 1)
    """
    np.random.seed(seed)
    
    # Generate market caps following a realistic power law distribution
    market_caps = np.random.pareto(1.5, n_assets) + 0.5
    weights = market_caps / market_caps.sum()
    
    return pd.Series(weights)

def analyze_returns(returns_df: pd.DataFrame) -> None:
    """
    Analyze and visualize the returns data.
    
    Args:
        returns_df: DataFrame of daily returns
    """
    print("=" * 70)
    print("RETURNS ANALYSIS")
    print("=" * 70)
    
    # Calculate key statistics
    annual_returns = returns_df.mean() * 252
    annual_vols = returns_df.std() * np.sqrt(252)
    sharpe = annual_returns / annual_vols
    
    print(f"Number of Assets: {len(returns_df.columns)}")
    print(f"Time Period: {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"Number of Trading Days: {len(returns_df)}")
    print("\n")
    print("Asset Statistics:")
    print("-" * 70)
    print(f"{'Asset':<12} {'Ann Return':<12} {'Ann Vol':<10} {'Sharpe':<8}")
    print("-" * 70)
    
    for asset in returns_df.columns:
        print(f"{asset:<12} {annual_returns[asset]:<12.2%} {annual_vols[asset]:<10.2%} {sharpe[asset]:<8.2f}")
    
    # Plot correlations heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(returns_df.corr(), cmap='RdBu', center=0, vmin=-1, vmax=1, square=True)
    plt.title("Asset Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    cumulative_returns = (1 + returns_df).cumprod()
    cumulative_returns.plot(legend=False, alpha=0.7)
    plt.title("Cumulative Asset Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

def main():
    """
    Run data preparation pipeline.
    """
    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION ENGINE - DAY 1: DATA PREPARATION")
    print("=" * 70)
    
    # Generate synthetic returns data
    print("\nGenerating synthetic asset returns...")
    returns_df = generate_synthetic_returns()
    
    # Analyze returns
    analyze_returns(returns_df)
    
    # Calculate expected returns
    expected_returns = calculate_expected_returns(returns_df, method='historical')
    
    # Calculate covariance matrix
    cov_matrix = calculate_covariance_matrix(returns_df, method='ledoit-wolf')
    
    # Generate market weights
    market_weights = generate_market_weights(len(returns_df.columns))
    
    # Calculate implied equilibrium returns
    implied_returns = calculate_implied_equilibrium_returns(cov_matrix, market_weights)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATA PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Assets: {len(returns_df.columns)}")
    print(f"Trading Days: {len(returns_df)}")
    print(f"Data Range: {returns_df.index[0]} to {returns_df.index[-1]}")
    print("\n")
    print("Market Capitalization Weights (Top 5):")
    print("-" * 50)
    print(market_weights.sort_values(ascending=False).head(5))
    print("\n")
    print("Expected Returns (Top 5):")
    print("-" * 50)
    print(expected_returns.sort_values(ascending=False).head(5))
    
    # Save data for use in subsequent scripts
    # In a real implementation, you would save to CSV files
    # returns_df.to_csv('data/asset_returns.csv')
    # expected_returns.to_csv('data/expected_returns.csv')
    # cov_matrix.to_csv('data/covariance_matrix.csv')
    
    print("\n" + "=" * 70)
    print("Data preparation complete. Ready for Day 2: Optimization.")
    print("=" * 70)

if __name__ == "__main__":
    main()
