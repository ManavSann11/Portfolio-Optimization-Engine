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
