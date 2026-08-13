# Portfolio Optimization Engine

A mean-variance portfolio optimizer built in Python using cvxpy, with Black-Litterman integration for blending investor views with market equilibrium. Allocates capital across 30+ assets with sector and position-size constraints. 

## Project Overview

Modern portfolio theory, introduced by Harry Markowitz, provides a framework for constructing portfolios that maximize expected return for a given level of risk. This project implements a practical portfolio optimization engine that extends the basic mean-variance framework with real-world constraints and advanced features. 

The engine combines:

- **Mean-variance optimization** with convex optimization (cvxpy)
- **Black-Litterman model** for incorporating investor views
- **Practical constraints** including sector limits, position sizing, and turnover constraints
- **Risk management** through covariance matrix estimation and risk budgeting

## Mathematical Background

### Mean-Variance Optimization

The core problem is to find the portfolio weights that minimize risk for a given target return:

$$ \text{Minimize} \quad \frac{1}{2} w^T \Sigma w $$

$$ \text{Subject to} \quad w^T \mu = \mu_{\text{target}} $$

$$ \quad \sum_i w_i = 1 $$

$$ \quad w_i \geq 0 $$

Where:
$w$ = vector of portfolio weights
- $\Sigma$ = covariance matrix of asset returns
- $\mu$ = vector of expected returns
- $\mu_{\text{target}}$ = target portfolio return

### Black-Litterman Model

The Black-Litterman model provides a systematic way to incorporate investor views into the expected return estimation. It combines: 

1. **Market equilibrium returns** derived from market capitalization weights
2. **Investor views** on specific assets or groups of assets
3. **Confidence levels** in those views

The Black-Litterman expected return is:

$$ E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q] $$

Where:
- $\Pi$ = implied equilibrium returns
- $\tau$ = scaling factor for uncertainty in equilibrium
- $P$ = matrix linking views to assets
- $\Omega$ = uncertainty in views
- $Q$ = vector of views (e.g., "Asset A will outperform Asset B by 2%")

### Efficient Frontier

The efficient frontier represents the set of optimal portfolios that offer the highest expected return for a given level of risk. Each point on the frontier corresponds to a different target return.

## Project Progression

### Day 1: Data Preparation and Covariance Estimation
Loads asset return data, calculates expected returns, and estimates the covariance matrix using methods including sample covariance and Ledoit-Wolf shrinkage.

### Day 2: Mean-Variance Optimization
Implements the basic mean-variance optimizer using cvxpy, with constraints including full investment, long-only positions, and sector limits.

### Day 3: Efficient Frontier
Computes the efficient frontier by solving the optimization problem across a range of target returns.

### Day 4: Black-Litterman Integration
Extends the model with the Black-Litterman framework for incorporating investor views and generating robust expected returns.

### Day 5: Constraint Implementation
Adds practical constraints including position size limits, sector exposure caps, and turnover constraints.

### Day 6: Portfolio Analysis and Visualization
Provides comprehensive analysis including performance metrics, risk decomposition, and visualization tools.

## Key Features

- **Convex Optimization**: Uses cvxpy for efficient, reliable optimization
- **Black-Litterman Integration**: Blends market equilibrium with investor views
- **Real-World Constraints**: Sector limits, position sizing, turnover constraints
- **Comprehensive Analysis**: Sharpe ratio, risk decomposition, efficient frontier
- **Visualization Tools**: Efficient frontier plots, allocation charts, risk decomposition

## Repository Structure
├── README.md
├── requirements.txt
├── .gitignore
├── 01_data_preparation.py
├── 02_mean_variance_optimization.py
├── 03_efficient_frontier.py
├── 04_black_litterman.py
├── 05_constraints.py
├── 06_portfolio_analysis.py

## Tech Stack

- Python 3
- cvxpy for convex optimization
- NumPy for numerical computations
- Pandas for data manipulation
- Matplotlib for visualization

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run scripts in order from 01 to 06

## Results

The optimizer produces portfolios that: 

- **Maximize Sharpe ratio**: Optimal risk-return trade-off
- **Respect constraints**: Sector limits, position sizes, and turnover
- **Incorporate views**: Black-Litterman blends market equilibrium with investor insights
- **Supports decision-making**: Clear visualizations and risk metrics

## Author
Manav Sannappanavar
NYU | Mathematics and Data Science
