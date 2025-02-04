import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import yfinance as yf

class MonteCarloValidator:
    def __init__(self, ticker='AAPL', years=10, simulations=10000):
        """
        Initialize Monte Carlo Validation with comprehensive checks
        
        Theoretical Foundations:
        1. Geometric Brownian Motion (GBM)
        2. Black-Scholes Model assumptions
        3. Statistical distribution validation
        """
        self.ticker = ticker
        self.years = years
        self.simulations = simulations
        
        # Fetch historical data
        self.stock_data = yf.Ticker(ticker).history(period=f"{years}y")
        
        # Calculate key statistical parameters
        self.close_prices = self.stock_data['Close']
        self.log_returns = np.log(self.close_prices / self.close_prices.shift(1)).dropna()
        
        # Core parameters
        self.mu = self.log_returns.mean()  # Expected return
        self.sigma = self.log_returns.std()  # Volatility
        self.last_price = self.close_prices.iloc[-1]
        
    def generate_simulations(self, risk_model='moderate'):
        """
        Generate Monte Carlo simulations using Geometric Brownian Motion
        
        Mathematical Model: dS = μS dt + σS dW
        Where:
        - dS: Change in stock price
        - μ: Drift (expected return)
        - σ: Volatility
        - dW: Wiener process (standard Brownian motion)
        """
        # Risk model adjustments
        risk_factors = {
            'conservative': 0.5,
            'moderate': 1.0,
            'aggressive': 1.5
        }
        
        # Adjust volatility based on risk model
        adjusted_sigma = self.sigma * risk_factors.get(risk_model, 1.0)
        
        # Trading days calculation
        trading_days = self.years * 252
        
        # Simulation using Geometric Brownian Motion
        np.random.seed(42)  # Reproducibility
        daily_returns = np.random.normal(
            (self.mu - 0.5 * adjusted_sigma**2), 
            adjusted_sigma, 
            (self.simulations, trading_days)
        )
        
        # Cumulative returns and price paths
        cumulative_returns = np.cumsum(daily_returns, axis=1)
        price_paths = self.last_price * np.exp(cumulative_returns)
        
        return price_paths
    
    def validate_distribution(self, simulations):
        """
        Validate simulation distribution against theoretical models
        
        Checks:
        1. Normality of log returns
        2. Comparison with historical log returns
        3. Statistical tests
        """
        # Calculate log returns from simulations
        sim_log_returns = np.log(simulations[:, 1:] / simulations[:, :-1])
        
        # Normality Tests
        _, p_value_shapiro = stats.shapiro(sim_log_returns.flatten())
        _, p_value_normaltest = stats.normaltest(sim_log_returns.flatten())
        
        # Descriptive Statistics Comparison
        validation_results = {
            'Historical Log Returns': {
                'Mean': self.log_returns.mean(),
                'Std Dev': self.log_returns.std(),
                'Skewness': self.log_returns.skew(),
                'Kurtosis': self.log_returns.kurtosis()
            },
            'Simulated Log Returns': {
                'Mean': np.mean(sim_log_returns),
                'Std Dev': np.std(sim_log_returns),
                'Skewness': stats.skew(sim_log_returns.flatten()),
                'Kurtosis': stats.kurtosis(sim_log_returns.flatten())
            },
            'Statistical Tests': {
                'Shapiro-Wilk p-value': p_value_shapiro,
                'D\'Agostino p-value': p_value_normaltest
            }
        }
        
        return validation_results
    
    def visualize_validation(self, simulations):
        """
        Create comprehensive visualization of simulation validation
        """
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Simulation Paths
        plt.subplot(2, 2, 1)
        plt.title('Monte Carlo Simulation Paths')
        for path in simulations[:50]:  # First 50 paths
            plt.plot(path, alpha=0.1, color='blue')
        plt.xlabel('Trading Days')
        plt.ylabel('Stock Price')
        
        # Subplot 2: Histogram of Final Prices
        plt.subplot(2, 2, 2)
        plt.title('Distribution of Final Prices')
        plt.hist(simulations[:, -1], bins=50, density=True, alpha=0.7)
        plt.xlabel('Final Stock Price')
        plt.ylabel('Density')
        
        # Subplot 3: Log Returns Comparison
        plt.subplot(2, 2, 3)
        plt.title('Log Returns Comparison')
        sim_log_returns = np.log(simulations[:, 1:] / simulations[:, :-1]).flatten()
        plt.hist(self.log_returns, bins=50, alpha=0.5, density=True, label='Historical')
        plt.hist(sim_log_returns, bins=50, alpha=0.5, density=True, label='Simulated')
        plt.legend()
        plt.xlabel('Log Returns')
        plt.ylabel('Density')
        
        # Subplot 4: Percentile Paths
        plt.subplot(2, 2, 4)
        plt.title('Percentile Paths')
        percentiles = [10, 50, 90]
        for p in percentiles:
            plt.plot(np.percentile(simulations, p, axis=0), label=f'{p}th Percentile')
        plt.legend()
        plt.xlabel('Trading Days')
        plt.ylabel('Stock Price')
        
        plt.tight_layout()
        plt.savefig('monte_carlo_validation.png')
        plt.close()
    
    def run_validation(self, risk_model='moderate'):
        """
        Comprehensive Monte Carlo Validation
        """
        print(f"Validating Monte Carlo Simulation for {self.ticker}")
        
        # Generate Simulations
        simulations = self.generate_simulations(risk_model)
        
        # Validate Distribution
        validation_results = self.validate_distribution(simulations)
        
        # Visualize Results
        self.visualize_validation(simulations)
        
        return validation_results

def main():
    # Validate for multiple stocks and risk models
    stocks = ['AAPL', 'GOOGL', 'MSFT']
    risk_models = ['conservative', 'moderate', 'aggressive']
    
    for stock in stocks:
        for model in risk_models:
            print(f"\n--- Validation for {stock} with {model} risk model ---")
            validator = MonteCarloValidator(ticker=stock)
            results = validator.run_validation(model)
            
            # Print detailed results
            for category, metrics in results.items():
                print(f"\n{category}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")

if __name__ == '__main__':
    main()
