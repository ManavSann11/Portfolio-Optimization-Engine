"""
Portfolio Optimization Engine - Day 6: Portfolio Analysis
Provides comprehensive analysis including performance metrics, risk decomposition, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic asset returns
n_assets = 30
n_days = 252 * 3
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
expected_returns = returns_df.mean() * 252
cov_matrix = returns_df.cov() * 252

# Generate optimal portfolio weights (simulated from Day 5)
np.random.seed(123)
optimal_weights = np.random.dirichlet(np.ones(n_assets) * 0.5)
optimal_weights = optimal_weights / optimal_weights.sum()

# Calculate portfolio returns
portfolio_returns = returns_df @ optimal_weights
cumulative_returns = (1 + portfolio_returns).cumprod()

# Calculate rolling statistics
rolling_window = 60
rolling_returns = portfolio_returns.rolling(window=rolling_window).mean() * 252
rolling_volatility = portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252)
rolling_sharpe = (rolling_returns - 0.02) / rolling_volatility

# Calculate drawdowns
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative / running_max) - 1
max_drawdown = drawdown.min()

# Calculate performance metrics
total_return = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]) - 1
annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
annualized_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (annualized_return - 0.02) / annualized_volatility
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

# Sortino ratio (downside deviation)
negative_returns = portfolio_returns[portfolio_returns < 0]
downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.01
sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0

# Risk decomposition
risk_contributions = optimal_weights * (cov_matrix @ optimal_weights)
risk_contributions = risk_contributions / np.sum(risk_contributions)

# Performance summary
print("=" * 70)
print("PORTFOLIO PERFORMANCE ANALYSIS")
print("=" * 70)
print(f"\nPERFORMANCE METRICS:")
print("-" * 50)
print(f"Total Return:                {total_return*100:.2f}%")
print(f"Annualized Return:           {annualized_return*100:.2f}%")
print(f"Annualized Volatility:       {annualized_volatility*100:.2f}%")
print(f"Sharpe Ratio:                {sharpe_ratio:.4f}")
print(f"Sortino Ratio:               {sortino_ratio:.4f}")
print(f"Calmar Ratio:                {calmar_ratio:.4f}")
print(f"Maximum Drawdown:            {max_drawdown*100:.2f}%")
print(f"Win Rate:                    {(portfolio_returns > 0).mean()*100:.2f}%")
print(f"Average Daily Return:        {portfolio_returns.mean()*100:.4f}%")
print(f"Positive Days:               {(portfolio_returns > 0).sum()}")
print(f"Negative Days:               {(portfolio_returns < 0).sum()}")

# Risk decomposition
print("\n" + "=" * 70)
print("RISK DECOMPOSITION")
print("=" * 70)
print(f"\n{'Asset':<12} {'Weight':<10} {'Risk Contribution':<18} {'% of Total Risk':<16}")
print("-" * 60)
sorted_idx = np.argsort(risk_contributions)[::-1]
for i in sorted_idx[:10]:
    print(f"{asset_names[i]:<12} {optimal_weights[i]:<10.2%} {risk_contributions[i]:<18.4f} {risk_contributions[i]/np.sum(risk_contributions)*100:<16.2f}%")

# Asset allocation
print("\n" + "=" * 70)
print("ASSET ALLOCATION")
print("=" * 70)
print(f"\nTop 10 Holdings:")
print("-" * 40)
sorted_weights = sorted([(asset, weight) for asset, weight in zip(asset_names, optimal_weights)], 
                       key=lambda x: x[1], reverse=True)
for asset, weight in sorted_weights[:10]:
    print(f"  {asset}: {weight*100:.2f}%")

# Plot cumulative returns with drawdowns
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Cumulative returns
axes[0].plot(cumulative_returns.index, cumulative_returns, linewidth=2)
axes[0].set_title('Portfolio Cumulative Returns')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Cumulative Return')
axes[0].grid(True)

# Plot 2: Drawdowns
axes[1].fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
axes[1].plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
axes[1].set_title('Portfolio Drawdowns')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Drawdown (%)')
axes[1].grid(True)
plt.tight_layout()
plt.show()

# Plot rolling statistics
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Rolling returns
axes[0].plot(rolling_returns.index, rolling_returns, linewidth=2, color='blue')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[0].set_title('Rolling Annualized Returns (60-day window)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Return')
axes[0].grid(True)

# Rolling volatility
axes[1].plot(rolling_volatility.index, rolling_volatility, linewidth=2, color='orange')
axes[1].set_title('Rolling Annualized Volatility (60-day window)')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volatility')
axes[1].grid(True)

# Rolling Sharpe ratio
axes[2].plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='green')
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[2].set_title('Rolling Sharpe Ratio (60-day window)')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Sharpe Ratio')
axes[2].grid(True)
plt.tight_layout()
plt.show()

# Plot returns distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of returns
axes[0].hist(portfolio_returns * 100, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[0].set_title('Daily Returns Distribution')
axes[0].set_xlabel('Daily Return (%)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True)

# QQ plot
from scipy import stats
stats.probplot(portfolio_returns, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Normal Distribution)')
axes[1].grid(True)
plt.tight_layout()
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = returns_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu', center=0, square=True, linewidths=0.5)
plt.title('Asset Correlation Matrix')
plt.tight_layout()
plt.show()

# Plot risk-return scatter
plt.figure(figsize=(10, 8))
asset_returns = expected_returns
asset_risks = [np.sqrt(cov_matrix.loc[asset, asset]) for asset in asset_names]
plt.scatter(asset_risks, asset_returns, alpha=0.6, s=50, label='Individual Assets')
for i, asset in enumerate(asset_names):
    if i % 5 == 0:
        plt.annotate(asset, (asset_risks[i], asset_returns[i]), fontsize=8)
port_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
port_return = expected_returns @ optimal_weights
plt.scatter(port_risk, port_return, color='red', s=150, marker='D', label='Optimal Portfolio')
plt.xlabel('Risk (σ)')
plt.ylabel('Expected Return')
plt.title('Risk-Return Scatter Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("Portfolio analysis complete.")
print("=" * 70)
