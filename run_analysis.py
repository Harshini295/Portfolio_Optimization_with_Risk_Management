from portfolio_optimizer import PortfolioOptimizer
def main():
    """Main execution function"""
    print("Portfolio Optimization with Risk Management")
    print("=" * 50)
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    config = {
        'use_synthetic': False,  # Set to False to use real Yahoo Finance data
        'min_weight': 0.05,     # 5% minimum allocation
        'max_weight': 0.40,     # 40% maximum allocation
        'n_simulations': 1000,  # Monte Carlo simulations
        'n_days': 252,          # 1 year projection
    }
    
    print(f"Configuration: {config}")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(symbols)
    
    # Run analysis
    try:
        results = optimizer.run_full_analysis(**config)
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Check the generated plots for visual analysis")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Required packages (install with: pip install numpy pandas matplotlib seaborn scipy yfinance)
    required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'yfinance']
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.optimize import minimize
        import yfinance as yf
        
        main()
        
    except ImportError as e:
        print(f"Missing required package: {e}")
        print(f"\nTo install all required packages, run:")
        print(f"pip install {' '.join(required_packages)}")