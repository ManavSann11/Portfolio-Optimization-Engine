"""
Portfolio Optimization Engine - Day 4: Black-Litterman Integration
Implements the Black-Litterman model to incorporate investor views.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

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

# Generate market capitalization weights (market equilibrium)
market_caps = np.random.pareto(1.5, n_assets) + 0.5
market_weights = market_caps / market_caps.sum()

# Risk aversion coefficient
lambda_risk = 2.5

# Implied equilibrium returns (reverse optimization)
implied_returns = lambda_risk * cov_matrix @ market_weights
implied_returns = pd.Series(implied_returns, index=asset_names)

# Black-Litterman parameters
tau = 0.05  # Uncertainty scaling factor for equilibrium returns

# Define investor views
# View 1: Asset_01 will outperform Asset_02 by 2%
P = np.array([
    [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
Q = np.array([0.02, -0.01])  # View returns

# View uncertainty matrix (confidence in views)
omega = np.array([
    [0.001, 0],
    [0, 0.001]
])

# Calculate Black-Litterman expected returns
tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
P_omega_inv = P.T @ np.linalg.inv(omega) @ P
combined_inv = tau_sigma_inv + P_omega_inv
combined = np.linalg.inv(combined_inv)

# Black-Litterman expected return
bl_returns = combined @ (tau_sigma_inv @ implied_returns + P.T @ np.linalg.inv(omega) @ Q)
bl_returns = pd.Series(bl_returns, index=asset_names)

# Posterior covariance matrix (optional)
bl_cov = combined + cov_matrix
bl_cov = pd.DataFrame(bl_cov, index=asset_names, columns=asset_names)

print("=" * 70)
print("BLACK-LITTERMAN MODEL RESULTS")
print("=" * 70)
print(f"\nRisk Aversion Coefficient (λ): {lambda_risk}")
print(f"Uncertainty Scaling Factor (τ): {tau}")
print("\nInvestor Views:")
print("  View 1: Asset_01 outperforms Asset_02 by 2.0%")
print("  View 2: Asset_03 underperforms Asset_04 by 1.0%")
print(f"  View Uncertainty (Ω): \n{omega}")
print("\n" + "-" * 50)
print("Implied Equilibrium Returns vs Black-Litterman Expected Returns")
print("-" * 50)
print(f"{'Asset':<12} {'Implied':<14} {'BL Return':<14} {'Difference':<12}")
print("-" * 50)
for asset in asset_names:
    diff = bl_returns[asset] - implied_returns[asset]
    print(f"{asset:<12} {implied_returns[asset]:<14.6f} {bl_returns[asset]:<14.6f} {diff:<12.6f}")

# Optimize using Black-Litterman returns
n_assets = len(bl_returns)
weights = cp.Variable(n_assets)
objective = cp.Maximize(bl_returns.values @ weights - (lambda_risk / 2) * cp.quad_form(weights, cov_matrix.values))
constraints = [cp.sum(weights) == 1, weights >= 0]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=False)
bl_optimal_weights = weights.value

# Compare with market portfolio
print("\n" + "=" * 70)
print("OPTIMAL PORTFOLIO COMPARISON")
print("=" * 70)
print(f"{'Asset':<12} {'Market Weight':<16} {'BL Optimal':<14}")
print("-" * 70)
for i, asset in enumerate(asset_names):
    print(f"{asset:<12} {market_weights[i]:<16.4f} {bl_optimal_weights[i]:<14.4f}")

# Calculate portfolio stats
bl_return = np.sum(bl_returns.values * bl_optimal_weights)
bl_risk = np.sqrt(bl_optimal_weights @ cov_matrix.values @ bl_optimal_weights)
bl_sharpe = bl_return / bl_risk if bl_risk > 0 else 0

market_return = np.sum(implied_returns * market_weights)
market_risk = np.sqrt(market_weights @ cov_matrix.values @ market_weights)
market_sharpe = market_return / market_risk if market_risk > 0 else 0

print("\n" + "-" * 50)
print("PORTFOLIO STATISTICS")
print("-" * 50)
print(f"{'Metric':<18} {'Market Portfolio':<20} {'Black-Litterman':<20}")
print("-" * 50)
print(f"{'Return':<18} {market_return:<20.4f} {bl_return:<20.4f}")
print(f"{'Risk':<18} {market_risk:<20.4f} {bl_risk:<20.4f}")
print(f"{'Sharpe':<18} {market_sharpe:<20.4f} {bl_sharpe:<20.4f}")

# Plot comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(asset_names))
width = 0.35
plt.bar(x - width/2, implied_returns.values, width, label='Implied Equilibrium Returns')
plt.bar(x + width/2, bl_returns.values, width, label='Black-Litterman Expected Returns')
plt.xlabel('Assets')
plt.ylabel('Expected Return')
plt.title('Implied Equilibrium vs Black-Litterman Expected Returns')
plt.xticks(x, asset_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
x = np.arange(len(asset_names))
width = 0.35
plt.bar(x - width/2, market_weights, width, label='Market Portfolio Weights')
plt.bar(x + width/2, bl_optimal_weights, width, label='Black-Litterman Optimal Weights')
plt.xlabel('Assets')
plt.ylabel('Portfolio Weight')
plt.title('Market Portfolio vs Black-Litterman Optimal Portfolio')
plt.xticks(x, asset_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("Black-Litterman analysis complete.")
print("=" * 70)
