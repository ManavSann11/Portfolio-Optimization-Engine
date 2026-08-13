"""
Portfolio Optimization Engine - Day 5: Constraints Implementation
Adds practical constraints including sector limits, position sizing, and turnover.
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

# Assign sectors to assets
sectors = ['Technology', 'Financials', 'Healthcare', 'Consumer', 'Energy']
asset_sectors = {}
sector_counts = {}
for i, asset in enumerate(asset_names):
    sector = np.random.choice(sectors)
    asset_sectors[asset] = sector
    sector_counts[sector] = sector_counts.get(sector, 0) + 1

# Create sector mapping arrays
sector_list = [asset_sectors[asset] for asset in asset_names]
unique_sectors = list(set(sector_list))
n_sectors = len(unique_sectors)

# Previous portfolio weights (for turnover constraint)
np.random.seed(123)
previous_weights = np.random.dirichlet(np.ones(n_assets))

# Optimization parameters
lambda_risk = 2.5
risk_free_rate = 0.02
max_position_size = 0.10  # Maximum 10% in any single asset
min_position_size = 0.01  # Minimum 1% in any asset (if included)
sector_min_weight = 0.05   # Minimum 5% per sector
sector_max_weight = 0.40   # Maximum 40% per sector
max_turnover = 0.20        # Maximum 20% turnover from previous weights

# Unconstrained optimization (baseline)
weights = cp.Variable(n_assets)
objective = cp.Maximize(expected_returns.values @ weights - (lambda_risk / 2) * cp.quad_form(weights, cov_matrix.values))
constraints = [cp.sum(weights) == 1, weights >= 0]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=False)
unconstrained_weights = weights.value

# Portfolio with position size constraints
weights = cp.Variable(n_assets)
objective = cp.Maximize(expected_returns.values @ weights - (lambda_risk / 2) * cp.quad_form(weights, cov_matrix.values))
constraints = [
    cp.sum(weights) == 1,
    weights >= 0,
    weights <= max_position_size,
    weights >= min_position_size  # Forces diversification
]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=False)
position_constrained_weights = weights.value

# Portfolio with sector constraints
sector_weights = {}
for sector in unique_sectors:
    sector_indices = [i for i, s in enumerate(sector_list) if s == sector]
    sector_weights[sector] = cp.sum(weights[sector_indices])

weights = cp.Variable(n_assets)
objective = cp.Maximize(expected_returns.values @ weights - (lambda_risk / 2) * cp.quad_form(weights, cov_matrix.values))
constraints = [
    cp.sum(weights) == 1,
    weights >= 0,
    sector_weights['Technology'] >= sector_min_weight,
    sector_weights['Technology'] <= sector_max_weight,
    sector_weights['Financials'] >= sector_min_weight,
    sector_weights['Financials'] <= sector_max_weight,
    sector_weights['Healthcare'] >= sector_min_weight,
    sector_weights['Healthcare'] <= sector_max_weight,
    sector_weights['Consumer'] >= sector_min_weight,
    sector_weights['Consumer'] <= sector_max_weight,
    sector_weights['Energy'] >= sector_min_weight,
    sector_weights['Energy'] <= sector_max_weight
]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=False)
sector_constrained_weights = weights.value

# Portfolio with turnover constraint
weights = cp.Variable(n_assets)
objective = cp.Maximize(expected_returns.values @ weights - (lambda_risk / 2) * cp.quad_form(weights, cov_matrix.values))
turnover = cp.sum(cp.abs(weights - previous_weights))
constraints = [
    cp.sum(weights) == 1,
    weights >= 0,
    turnover <= max_turnover
]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=False)
turnover_constrained_weights = weights.value

# Portfolio with all constraints combined
weights = cp.Variable(n_assets)
sector_weights = {}
for sector in unique_sectors:
    sector_indices = [i for i, s in enumerate(sector_list) if s == sector]
    sector_weights[sector] = cp.sum(weights[sector_indices])

turnover = cp.sum(cp.abs(weights - previous_weights))
objective = cp.Maximize(expected_returns.values @ weights - (lambda_risk / 2) * cp.quad_form(weights, cov_matrix.values))
constraints = [
    cp.sum(weights) == 1,
    weights >= 0,
    weights <= max_position_size,
    weights >= min_position_size,
    sector_weights['Technology'] >= sector_min_weight,
    sector_weights['Technology'] <= sector_max_weight,
    sector_weights['Financials'] >= sector_min_weight,
    sector_weights['Financials'] <= sector_max_weight,
    sector_weights['Healthcare'] >= sector_min_weight,
    sector_weights['Healthcare'] <= sector_max_weight,
    sector_weights['Consumer'] >= sector_min_weight,
    sector_weights['Consumer'] <= sector_max_weight,
    sector_weights['Energy'] >= sector_min_weight,
    sector_weights['Energy'] <= sector_max_weight,
    turnover <= max_turnover
]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS, verbose=False)
all_constrained_weights = weights.value

# Calculate portfolio statistics
def calculate_portfolio_stats(weights):
    port_return = np.sum(expected_returns.values * weights)
    port_risk = np.sqrt(weights @ cov_matrix.values @ weights)
    sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
    return port_return, port_risk, sharpe

stats = {}
portfolios = {
    'Unconstrained': unconstrained_weights,
    'Position Limits': position_constrained_weights,
    'Sector Limits': sector_constrained_weights,
    'Turnover Limit': turnover_constrained_weights,
    'All Constraints': all_constrained_weights
}

print("=" * 70)
print("CONSTRAINT IMPLEMENTATION RESULTS")
print("=" * 70)
print(f"\nParameters:")
print(f"  Max Position Size: {max_position_size*100:.0f}%")
print(f"  Min Position Size: {min_position_size*100:.0f}%")
print(f"  Sector Min Weight: {sector_min_weight*100:.0f}%")
print(f"  Sector Max Weight: {sector_max_weight*100:.0f}%")
print(f"  Max Turnover: {max_turnover*100:.0f}%")
print(f"\nSector Assignments:")
for sector, count in sector_counts.items():
    print(f"  {sector}: {count} assets")

print("\n" + "-" * 70)
print("PORTFOLIO STATISTICS COMPARISON")
print("-" * 70)
print(f"{'Portfolio':<18} {'Return':<12} {'Risk':<12} {'Sharpe':<12} {'Active Weight':<12}")
print("-" * 70)

for name, w in portfolios.items():
    ret, risk, sharpe = calculate_portfolio_stats(w)
    if name != 'Unconstrained':
        tracking_error = np.sqrt(np.sum((w - unconstrained_weights)**2))
    else:
        tracking_error = 0
    print(f"{name:<18} {ret:<12.4f} {risk:<12.4f} {sharpe:<12.4f} {tracking_error:<12.4f}")

# Print top holdings
print("\n" + "-" * 70)
print("TOP 5 HOLDINGS BY PORTFOLIO")
print("-" * 70)
for name, w in portfolios.items():
    top_idx = np.argsort(w)[-5:][::-1]
    top_assets = [asset_names[i] for i in top_idx]
    top_weights = [w[i] for i in top_idx]
    print(f"\n{name}:")
    for asset, weight in zip(top_assets, top_weights):
        print(f"  {asset}: {weight*100:.2f}%")

# Plot weights comparison
plt.figure(figsize=(14, 10))
x = np.arange(n_assets)
width = 0.15
offset = 0

for i, (name, w) in enumerate(portfolios.items()):
    plt.bar(x + offset, w, width, label=name)
    offset += width

plt.xlabel('Assets')
plt.ylabel('Portfolio Weight')
plt.title('Portfolio Weights Comparison Across Constraint Sets')
plt.xticks(x + width*2, asset_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot sector allocation
plt.figure(figsize=(14, 8))
sector_data = {}
for name, w in portfolios.items():
    sector_alloc = {}
    for sector in unique_sectors:
        sector_indices = [i for i, s in enumerate(sector_list) if s == sector]
        sector_alloc[sector] = np.sum(w[sector_indices])
    sector_data[name] = sector_alloc

df_sectors = pd.DataFrame(sector_data)
ax = df_sectors.plot(kind='bar', figsize=(12, 6))
plt.xlabel('Sector')
plt.ylabel('Portfolio Weight')
plt.title('Sector Allocation Across Constraint Sets')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("Constraint implementation complete.")
print("=" * 70)
