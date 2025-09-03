#!/usr/bin/env python3
"""
Portfolio Optimization with Risk Management
Advanced portfolio construction using Markowitz optimization, Monte Carlo simulations, and comprehensive risk analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    def __init__(self, symbols, start_date=None, end_date=None):
        """
        Initialize Portfolio Optimizer
        
        Args:
            symbols (list): List of stock symbols
            start_date (str): Start date for data (YYYY-MM-DD)
            end_date (str): End date for data (YYYY-MM-DD)
        """
        self.symbols = symbols
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.mean_returns = None
        
    def fetch_data(self, use_synthetic=False):
        """Fetch stock price data"""
        if use_synthetic:
            print("Generating synthetic data for demonstration...")
            self._generate_synthetic_data()
        else:
            print(f"Fetching data for {self.symbols} from {self.start_date} to {self.end_date}...")
            try:
                self.data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Adj Close']
                if self.data.empty:
                    raise ValueError("No data fetched")
                print(f"Successfully fetched data for {len(self.data.columns)} assets")
            except Exception as e:
                print(f"Error fetching data: {e}")
                print("Falling back to synthetic data...")
                self._generate_synthetic_data()
        
        self._calculate_returns()
    
    def _generate_synthetic_data(self):
        """Generate synthetic stock price data for demonstration"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Remove weekends
        
        data = {}
        for symbol in self.symbols:
            # Random walk with drift
            np.random.seed(hash(symbol) % 2**32)  
            
            initial_price = 100 + np.random.uniform(0, 400)
            volatility = 0.15 + np.random.uniform(0, 0.25)  # 15-40% annual volatility
            drift = 0.05 + np.random.uniform(0, 0.15)  # 5-20% annual return
            
            dt = 1/252
            n_days = len(dates)
            
            returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_days)
            prices = [initial_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            data[symbol] = prices[1:]  
        
        self.data = pd.DataFrame(data, index=dates)
    
    def _calculate_returns(self):
        """Calculate daily returns and statistics"""
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252  # Annualized
        
        print(f"\nAnnualized Statistics:")
        print(f"{'Asset':<8} {'Return':<8} {'Volatility':<12} {'Sharpe':<8}")
        print("-" * 40)
        
        for symbol in self.symbols:
            ret = self.mean_returns[symbol] * 100
            vol = np.sqrt(self.cov_matrix.loc[symbol, symbol]) * 100
            sharpe = self.mean_returns[symbol] / np.sqrt(self.cov_matrix.loc[symbol, symbol])
            print(f"{symbol:<8} {ret:>6.2f}% {vol:>10.2f}% {sharpe:>6.3f}")
    
    def portfolio_metrics(self, weights):
        """Calculate portfolio metrics given weights"""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': sharpe_ratio,
            'variance': portfolio_variance
        }
    
    def optimize_portfolio(self, objective='max_sharpe', target_return=None, 
                          min_weight=0.05, max_weight=0.4):
        """
        Optimize portfolio using Mean-Variance Optimization
        
        Args:
            objective (str): 'max_sharpe', 'min_risk', or 'target_return'
            target_return (float): Target return for 'target_return' objective
            min_weight (float): Minimum weight per asset
            max_weight (float): Maximum weight per asset
        """
        n_assets = len(self.symbols)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if objective == 'target_return' and target_return:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(self.mean_returns * x) - target_return
            })
        
        # Bounds for each weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Objective functions
        def neg_sharpe(weights):
            metrics = self.portfolio_metrics(weights)
            return -metrics['sharpe'] 
        
        def portfolio_volatility(weights):
            return self.portfolio_metrics(weights)['volatility']
        
        # Choose objective function
        if objective == 'max_sharpe':
            obj_func = neg_sharpe
        else:  # min_risk or target_return
            obj_func = portfolio_volatility
        
        # Optimize
        result = minimize(
            obj_func, 
            initial_guess, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            metrics = self.portfolio_metrics(optimal_weights)
            
            return {
                'weights': optimal_weights,
                'metrics': metrics,
                'optimization_result': result
            }
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    def generate_efficient_frontier(self, n_points=50, min_weight=0.05, max_weight=0.4):
        """Generate efficient frontier"""
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                result = self.optimize_portfolio(
                    objective='target_return',
                    target_return=target_ret,
                    min_weight=min_weight,
                    max_weight=max_weight
                )
                efficient_portfolios.append({
                    'return': result['metrics']['return'],
                    'volatility': result['metrics']['volatility'],
                    'sharpe': result['metrics']['sharpe'],
                    'weights': result['weights']
                })
            except:
                continue  # Skip infeasible points
        
        return efficient_portfolios
    
    def monte_carlo_simulation(self, weights, n_simulations=1000, n_days=252, initial_value=100000):
        """Run Monte Carlo simulation for portfolio"""
        print(f"\nRunning {n_simulations} Monte Carlo simulations over {n_days} days...")
        
        # Portfolio historical returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        simulations = []
        
        for i in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, n_days)
            
            # Calculate portfolio value path
            values = [initial_value]
            for ret in random_returns:
                values.append(values[-1] * (1 + ret))
            
            # Calculate metrics for this simulation
            max_drawdown = self._calculate_max_drawdown(values)
            final_value = values[-1]
            total_return = (final_value - initial_value) / initial_value
            
            simulations.append({
                'final_value': final_value,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'path': values
            })
        
        return simulations
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown from value series"""
        peak = values[0]
        max_dd = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_risk_metrics(self, simulations, confidence_level=0.95):
        """Calculate VaR, CVaR, and other risk metrics"""
        returns = [sim['total_return'] for sim in simulations]
        final_values = [sim['final_value'] for sim in simulations]
        max_drawdowns = [sim['max_drawdown'] for sim in simulations]
        
        # Sort returns for percentile calculations
        sorted_returns = sorted(returns)
        
        # Value at Risk
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[var_index] * 100  # Convert to positive percentage
        
        # Conditional VaR (Expected Shortfall)
        cvar = -np.mean(sorted_returns[:var_index + 1]) * 100
        
        # Other metrics
        avg_max_drawdown = np.mean(max_drawdowns) * 100
        worst_case = min(final_values)
        best_case = max(final_values)
        success_prob = len([r for r in returns if r > 0]) / len(returns) * 100
        
        return {
            'var': var,
            'cvar': cvar,
            'avg_max_drawdown': avg_max_drawdown,
            'worst_case': worst_case,
            'best_case': best_case,
            'success_probability': success_prob,
            'return_distribution': sorted_returns
        }
    
    def plot_results(self, optimal_portfolio, efficient_frontier, risk_metrics, simulations):
        """Create comprehensive visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Efficient Frontier
        if efficient_frontier:
            risks = [p['volatility'] * 100 for p in efficient_frontier]
            returns = [p['return'] * 100 for p in efficient_frontier]
            sharpes = [p['sharpe'] for p in efficient_frontier]
            
            scatter = ax1.scatter(risks, returns, c=sharpes, cmap='viridis', alpha=0.7)
            ax1.scatter(optimal_portfolio['metrics']['volatility'] * 100, 
                       optimal_portfolio['metrics']['return'] * 100,
                       color='red', s=100, marker='*', label='Optimal Portfolio')
            ax1.set_xlabel('Risk (Volatility %)')
            ax1.set_ylabel('Expected Return %')
            ax1.set_title('Efficient Frontier')
            ax1.legend()
            plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
        
        # 2. Portfolio Allocation
        weights = optimal_portfolio['weights']
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.symbols)))
        wedges, texts, autotexts = ax2.pie(weights, labels=self.symbols, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Optimal Portfolio Allocation')
        
        # 3. Return Distribution
        returns_dist = [sim['total_return'] * 100 for sim in simulations]
        ax3.hist(returns_dist, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(risk_metrics['var'], color='red', linestyle='--', 
                   label=f'VaR (95%): -{risk_metrics["var"]:.2f}%')
        ax3.axvline(risk_metrics['cvar'], color='darkred', linestyle='--',
                   label=f'CVaR (95%): -{risk_metrics["cvar"]:.2f}%')
        ax3.set_xlabel('Total Return %')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Monte Carlo Return Distribution')
        ax3.legend()
        
        # 4. Sample Portfolio Paths
        n_paths_to_show = min(100, len(simulations))
        for i in range(n_paths_to_show):
            path = simulations[i]['path']
            days = range(len(path))
            alpha = 0.1 if i < n_paths_to_show - 1 else 1.0
            color = 'lightblue' if i < n_paths_to_show - 1 else 'red'
            linewidth = 0.5 if i < n_paths_to_show - 1 else 2
            ax4.plot(days, path, color=color, alpha=alpha, linewidth=linewidth)
        
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Portfolio Value ($)')
        ax4.set_title(f'Monte Carlo Simulation Paths (showing {n_paths_to_show} of {len(simulations)})')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self, optimal_portfolio, risk_metrics):
        """Print detailed results"""
        print("\n" + "="*80)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("="*80)
        
        # Portfolio Metrics
        metrics = optimal_portfolio['metrics']
        print(f"\nOPTIMAL PORTFOLIO METRICS:")
        print(f"Expected Annual Return: {metrics['return']*100:>8.2f}%")
        print(f"Annual Volatility:      {metrics['volatility']*100:>8.2f}%")
        print(f"Sharpe Ratio:           {metrics['sharpe']:>8.3f}")
        
        # Allocation
        print(f"\nPORTFOLIO ALLOCATION:")
        print(f"{'Asset':<8} {'Weight':<8} {'Value ($100K)':<12} {'Expected Return':<15}")
        print("-" * 50)
        
        for i, symbol in enumerate(self.symbols):
            weight = optimal_portfolio['weights'][i]
            value = weight * 100000
            exp_ret = self.mean_returns[symbol] * 100
            print(f"{symbol:<8} {weight*100:>6.1f}% {value:>10,.0f} {exp_ret:>13.2f}%")
        
        # Risk Metrics
        print(f"\nRISK ANALYSIS (Monte Carlo - {len(risk_metrics.get('return_distribution', []))} simulations):")
        print(f"Value at Risk (95%):           -{risk_metrics['var']:>6.2f}%")
        print(f"Conditional VaR (95%):         -{risk_metrics['cvar']:>6.2f}%")
        print(f"Average Maximum Drawdown:      {risk_metrics['avg_max_drawdown']:>6.2f}%")
        print(f"Success Probability:           {risk_metrics['success_probability']:>6.1f}%")
        print(f"Worst Case Scenario:           ${risk_metrics['worst_case']:>8,.0f}")
        print(f"Best Case Scenario:            ${risk_metrics['best_case']:>8,.0f}")
        
        # Diversification Metrics
        weights = optimal_portfolio['weights']
        hhi = np.sum(weights**2)  # Herfindahl-Hirschman Index
        diversification_ratio = 1 - hhi
        
        print(f"\nDIVERSIFICATION ANALYSIS:")
        print(f"Diversification Ratio:         {diversification_ratio:>6.3f}")
        print(f"Effective Number of Assets:    {1/hhi:>6.1f}")
        print(f"Maximum Single Weight:         {np.max(weights)*100:>6.1f}%")
        print(f"Minimum Single Weight:         {np.min(weights)*100:>6.1f}%")
    
    def stress_test(self, weights, scenarios=None):
        """Perform stress testing on the portfolio"""
        if scenarios is None:
            scenarios = {
                'Market Crash': {'all': -0.30},  # 30% drop across all assets
                'Tech Selloff': {'AAPL': -0.40, 'GOOGL': -0.40, 'MSFT': -0.35, 'default': -0.10},
                'Interest Rate Shock': {'default': -0.15},  # 15% drop
                'Inflation Spike': {'default': -0.20}  # 20% drop
            }
        
        print(f"\nSTRESS TESTING RESULTS:")
        print(f"{'Scenario':<20} {'Portfolio Loss':<15} {'New Value ($100K)':<15}")
        print("-" * 52)
        
        for scenario_name, shocks in scenarios.items():
            portfolio_shock = 0
            
            for i, symbol in enumerate(self.symbols):
                if symbol in shocks:
                    asset_shock = shocks[symbol]
                elif 'all' in shocks:
                    asset_shock = shocks['all']
                else:
                    asset_shock = shocks.get('default', 0)
                
                portfolio_shock += weights[i] * asset_shock
            
            new_value = 100000 * (1 + portfolio_shock)
            loss_pct = -portfolio_shock * 100
            
            print(f"{scenario_name:<20} {loss_pct:>6.2f}% {new_value:>13,.0f}")
    
    def run_full_analysis(self, use_synthetic=False, min_weight=0.05, max_weight=0.4, 
                         n_simulations=1000, n_days=252):
        """Run complete portfolio optimization analysis"""
        print("Starting Portfolio Optimization Analysis...")
        print(f"Assets: {', '.join(self.symbols)}")
        
        # Fetch and prepare data
        self.fetch_data(use_synthetic=use_synthetic)
        
        # Optimize portfolio
        print("\nOptimizing portfolio (Markowitz Mean-Variance)...")
        optimal_portfolio = self.optimize_portfolio(
            objective='max_sharpe',
            min_weight=min_weight,
            max_weight=max_weight
        )
        
        # Generate efficient frontier
        print("Generating efficient frontier...")
        efficient_frontier = self.generate_efficient_frontier(
            min_weight=min_weight, 
            max_weight=max_weight
        )
        
        # Monte Carlo simulation
        simulations = self.monte_carlo_simulation(
            optimal_portfolio['weights'],
            n_simulations=n_simulations,
            n_days=n_days
        )
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(simulations)
        
        # Stress testing
        self.stress_test(optimal_portfolio['weights'])
        
        # Print results
        self.print_results(optimal_portfolio, risk_metrics)
        
        # Create visualizations
        print(f"\nGenerating visualizations...")
        self.plot_results(optimal_portfolio, efficient_frontier, risk_metrics, simulations)
        
        return {
            'optimal_portfolio': optimal_portfolio,
            'efficient_frontier': efficient_frontier,
            'risk_metrics': risk_metrics,
            'simulations': simulations
        }