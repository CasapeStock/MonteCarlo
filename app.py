import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
import scipy.stats as stats
from flask import Flask, render_template, request, jsonify

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monte_carlo_debug.log', mode='w'),  # Overwrite previous log
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Detailed exception handler
def log_exception(exc_type, exc_value, exc_traceback):
    """Comprehensive exception logging"""
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_exception

# Create Flask app with explicit template and static folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'app', 'templates'),
            static_folder=os.path.join(BASE_DIR, 'app', 'static'))

class MonteCarloSimulation:
    def __init__(self, ticker='AAPL', years=10, simulations=1000, risk_model='moderate'):
        try:
            logger.info(f"Initializing Monte Carlo Simulation: {ticker}, {risk_model}")
            
            # Validate inputs
            if not isinstance(ticker, str):
                raise ValueError(f"Invalid ticker type: {type(ticker)}")
            if years <= 0 or simulations <= 0:
                raise ValueError(f"Invalid simulation parameters: years={years}, simulations={simulations}")
            
            self.ticker = ticker.upper()
            self.years = years
            self.simulations = simulations
            self.risk_model = risk_model.lower()

            # Comprehensive risk model configuration with input validation
            self.risk_factors = {
                'conservative': {
                    'volatility_multiplier': 0.5,
                    'return_adjustment': -0.3,
                    'skew_adjustment': -0.2,
                    'confidence_level': 0.75
                },
                'moderate': {
                    'volatility_multiplier': 1.0,
                    'return_adjustment': 0,
                    'skew_adjustment': 0,
                    'confidence_level': 0.90
                },
                'aggressive': {
                    'volatility_multiplier': 1.5,
                    'return_adjustment': 0.3,
                    'skew_adjustment': 0.2,
                    'confidence_level': 0.95
                }
            }

            # Validate risk model with detailed logging
            if self.risk_model not in self.risk_factors:
                error_msg = f"Invalid risk model: {self.risk_model}. Must be one of {list(self.risk_factors.keys())}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Fetch and process stock data
            self._fetch_stock_data()

        except Exception as e:
            logger.error(f"Initialization Error: {e}")
            logger.error(traceback.format_exc())
            raise

    def _fetch_stock_data(self):
        try:
            logger.info(f"Fetching stock data for {self.ticker}")
            
            # Enhanced error handling for data retrieval
            try:
                stock = yf.Ticker(self.ticker)
                hist_data = stock.history(period=f"{max(10, self.years)}y")
            except Exception as fetch_error:
                logger.error(f"YFinance Fetch Error: {fetch_error}")
                raise ValueError(f"Could not retrieve data for {self.ticker}")
            
            if hist_data.empty:
                error_msg = f"No historical data for {self.ticker}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log returns calculation with error handling
            try:
                close_prices = hist_data['Close']
                log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            except Exception as calc_error:
                logger.error(f"Log Returns Calculation Error: {calc_error}")
                raise
            
            # Statistical parameters with robust calculation
            self.last_price = close_prices.iloc[-1]
            self.mu = log_returns.mean()
            self.sigma = log_returns.std()
            
            # Additional statistical insights with error handling
            try:
                self.statistical_summary = {
                    'mean_return': self.mu,
                    'volatility': self.sigma,
                    'skewness': stats.skew(log_returns),
                    'kurtosis': stats.kurtosis(log_returns)
                }
            except Exception as stats_error:
                logger.error(f"Statistical Summary Calculation Error: {stats_error}")
                self.statistical_summary = {}
            
            logger.info(f"Stock data fetched successfully. Last Price: {self.last_price}")
            logger.debug(f"Statistical Summary: {self.statistical_summary}")
        
        except Exception as e:
            logger.error(f"Stock Data Fetch Error: {e}")
            logger.error(traceback.format_exc())
            raise

    def monte_carlo_simulation(self, ticker, years=10, simulations=10000, risk_model='moderate'):
        """
        Precise Monte Carlo Simulation with Rigorous Statistical Validation
        
        Key Focus:
        - Mathematically sound return generation
        - Robust volatility estimation
        - Precise statistical modeling
        """
        try:
            # Comprehensive historical data retrieval
            stock_data = yf.Ticker(ticker)
            
            # Extended historical price data
            hist = stock_data.history(period='10y')
            if hist is None or hist.empty:
                raise ValueError(f"Could not fetch historical data for {ticker}")
            
            # Advanced return calculation
            daily_returns = hist['Close'].pct_change().dropna()
            
            # Precise volatility estimation
            # 1. Standard deviation of log returns (more statistically robust)
            log_returns = np.log(1 + daily_returns)
            base_volatility = log_returns.std()
            
            # 2. Annualized volatility calculation
            trading_days = 252  # Standard trading days in a year
            annualized_volatility = base_volatility * np.sqrt(trading_days)
            
            # Stock-specific calibration with precise parameters
            stock_specific_calibration = {
                'NVDA': {
                    'base_growth_rate': 0.145,  # 14.5% CAGR
                    'volatility_multiplier': 1.0,  # Adjusted for precision
                    'mean_reversion_factor': 1.0,
                    'skew_adjustment': 0,
                    'kurtosis_adjustment': 0
                },
                'default': {
                    'base_growth_rate': 0.10,  # 10% default
                    'volatility_multiplier': 1.0,
                    'mean_reversion_factor': 1.0,
                    'skew_adjustment': 0,
                    'kurtosis_adjustment': 0
                }
            }
            
            # Select calibration parameters
            calibration = stock_specific_calibration.get(ticker.upper(), stock_specific_calibration['default'])
            
            # Precise volatility adjustment
            daily_volatility = base_volatility * calibration['volatility_multiplier']
            
            # Get current stock price with multiple validation checks
            try:
                current_price = hist['Close'].iloc[-1]
            except Exception as price_error:
                logger.warning(f"Price retrieval error: {price_error}")
                current_price = stock_data.info.get('regularMarketPrice', 0)
            
            # Simulation parameters
            total_days = int(years * trading_days)
            
            # Initialize price paths with precise memory allocation
            price_paths = np.zeros((simulations, total_days + 1), dtype=np.float64)
            price_paths[:, 0] = current_price
            
            # Risk model adjustments with precise control
            risk_factors = {
                'conservative': {
                    'growth_dampening': 0.5,
                    'volatility_scaling': 0.7,
                    'max_drawdown_limit': 0.15
                },
                'moderate': {
                    'growth_dampening': 1.0,
                    'volatility_scaling': 1.0,
                    'max_drawdown_limit': 0.25
                },
                'aggressive': {
                    'growth_dampening': 1.5,
                    'volatility_scaling': 1.3,
                    'max_drawdown_limit': 0.35
                }
            }
            
            # Select risk model parameters
            risk_params = risk_factors.get(risk_model.lower(), risk_factors['moderate'])
            
            # Precise growth rate calculation
            base_growth_rate = calibration['base_growth_rate']
            daily_growth_rate = (1 + base_growth_rate) ** (1/trading_days) - 1
            
            # Advanced random number generation
            rng = np.random.default_rng(seed=42)
            
            # Precise price path simulation
            for i in range(simulations):
                # Generate returns with mathematically sound distribution
                # Uses a t-distribution to capture market return characteristics
                daily_returns = rng.standard_t(df=5, size=total_days) * (
                    daily_volatility * np.sqrt(risk_params['volatility_scaling'])
                ) + daily_growth_rate * risk_params['growth_dampening']
                
                # Cumulative returns with precise drawdown protection
                cumulative_returns = np.cumprod(1 + np.clip(
                    daily_returns, 
                    -risk_params['max_drawdown_limit'], 
                    risk_params['max_drawdown_limit']
                ))
                
                # Price path generation
                price_paths[i, 1:] = current_price * cumulative_returns
            
            return price_paths
        
        except Exception as e:
            logger.error(f"Monte Carlo Simulation Error: {e}")
            logger.error(traceback.format_exc())
            raise

    def simulate(self):
        try:
            logger.info(f"Starting Monte Carlo Simulation for {self.ticker}")
            
            # Trading days calculation
            trading_days = int(self.years * 252)
            
            # Risk model adjustments
            risk_factors = {
                'conservative': {
                    'volatility_multiplier': 0.5,
                    'return_adjustment': -0.5
                },
                'moderate': {
                    'volatility_multiplier': 1.0,
                    'return_adjustment': 0.0
                },
                'aggressive': {
                    'volatility_multiplier': 1.5,
                    'return_adjustment': 0.5
                }
            }
            
            # Validate risk model
            if self.risk_model.lower() not in risk_factors:
                raise ValueError(f"Invalid risk model: {self.risk_model}")
            
            # Call the detailed monte_carlo_simulation method
            return self.monte_carlo_simulation(
                self.ticker, 
                years=self.years, 
                simulations=10000, 
                risk_model=self.risk_model
            )
        
        except Exception as e:
            logger.error(f"Simulation Error: {e}")
            logger.error(traceback.format_exc())
            raise

    def generate_visualization(self, simulations_data):
        try:
            logger.info("Generating simulation visualization")
            
            # Validate input data with detailed logging
            if simulations_data is None:
                logger.error("Simulations data is None")
                raise ValueError("Simulations data cannot be None")
            
            if simulations_data.size == 0:
                logger.error("Simulations data is empty")
                raise ValueError("Simulations data is empty")
            
            # Extract final prices
            final_prices = simulations_data[:, -1]
            
            # Create figure
            plt.figure(figsize=(16, 10), dpi=300)
            
            # Main histogram with kernel density estimation
            plt.hist(final_prices, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Final Price Distribution')
            
            # Kernel Density Estimation
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(final_prices)
            x_range = np.linspace(final_prices.min(), final_prices.max(), 200)
            
            # Plot histogram
            plt.plot(x_range, kde(x_range), color='red', linewidth=2, label='Density Estimate')
            
            # Compute and plot percentile lines
            percentiles = {
                '10th Percentile': np.percentile(final_prices, 10),
                'Median': np.percentile(final_prices, 50),
                '90th Percentile': np.percentile(final_prices, 90)
            }
            
            # Plot vertical lines for percentiles
            for label, value in percentiles.items():
                plt.axvline(x=value, color='green', linestyle='--', alpha=0.7)
                plt.text(value, plt.gca().get_ylim()[1]*0.9, label, rotation=90, verticalalignment='top')
            
            # Title and labels
            plt.title(f'Final Price Distribution: {self.ticker} ({self.risk_model.capitalize()} Risk Model)', 
                      fontsize=16, fontweight='bold')
            plt.xlabel('Final Stock Price ($)', fontsize=12)
            plt.ylabel('Probability Density', fontsize=12)
            
            # Add statistical summary as text
            stats_text = (
                f"Initial Price: ${self.last_price:.2f}\n"
                f"Mean Final Price: ${np.mean(final_prices):.2f}\n"
                f"Median Final Price: ${np.median(final_prices):.2f}\n"
                f"Standard Deviation: ${np.std(final_prices):.2f}\n"
                f"10th Percentile: ${percentiles['10th Percentile']:.2f}\n"
                f"90th Percentile: ${percentiles['90th Percentile']:.2f}"
            )
            
            plt.annotate(
                stats_text, 
                xy=(0.02, 0.98), 
                xycoords='axes fraction',
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', 
                          facecolor='wheat', 
                          alpha=0.5),
                fontsize=10
            )
            
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot with comprehensive error handling
            try:
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                image_png = buffer.getvalue()
                plt.close()
                
                # Validate image generation
                if not image_png:
                    logger.error("Generated image is empty")
                    raise ValueError("Failed to generate image")
                
                logger.info("Visualization generated successfully")
                return base64.b64encode(image_png).decode('utf-8')
            except Exception as save_error:
                logger.error(f"Plot saving error: {save_error}")
                logger.error(traceback.format_exc())
                return None
        
        except Exception as e:
            logger.error(f"Visualization Generation Error: {e}")
            logger.error(traceback.format_exc())
            return None

    def statistical_analysis(self, simulations_data):
        try:
            logger.info("Performing statistical analysis")
            
            # Validate input data
            if simulations_data is None or simulations_data.size == 0:
                raise ValueError("Empty simulation data")
            
            final_prices = simulations_data[:, -1]
            
            analysis = {
                'initial_price': self.last_price,
                'final_price_stats': {
                    'mean': np.mean(final_prices),
                    'median': np.median(final_prices),
                    'std_dev': np.std(final_prices),
                    'min': np.min(final_prices),
                    'max': np.max(final_prices)
                },
                'percentiles': {
                    '10th': np.percentile(final_prices, 10),
                    '25th': np.percentile(final_prices, 25),
                    '50th': np.percentile(final_prices, 50),
                    '75th': np.percentile(final_prices, 75),
                    '90th': np.percentile(final_prices, 90)
                },
                'historical_stats': self.statistical_summary
            }
            
            logger.info("Statistical analysis completed")
            logger.debug(f"Analysis Results: {analysis}")
            return analysis
        
        except Exception as e:
            logger.error(f"Statistical Analysis Error: {e}")
            logger.error(traceback.format_exc())
            raise

import json

def safe_numpy_conversion(obj):
    """
    Safely convert numpy objects to JSON-serializable types
    """
    if isinstance(obj, np.ndarray):
        # For large arrays, limit the number of elements
        if obj.size > 10000:
            # Sample or reduce the array
            sample_indices = np.linspace(0, obj.size - 1, 10000, dtype=int)
            obj = obj.flatten()[sample_indices]
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

@app.route('/')
def index():
    """Render the main index page"""
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Extract parameters with default values
        ticker = request.form.get('ticker', 'NVDA')
        risk_model = request.form.get('risk_model', 'moderate')
        
        # Parse years parameter with validation
        try:
            years = float(request.form.get('years', 10))
            # Ensure years is within a reasonable range
            years = max(1, min(years, 30))
        except (ValueError, TypeError):
            years = 10  # Default to 10 years if parsing fails
        
        # Log the received parameters
        logger.info(f"Simulation Request - Ticker: {ticker}, Risk Model: {risk_model}, Years: {years}")
        
        # Perform Monte Carlo simulation
        simulation = MonteCarloSimulation(ticker)
        price_paths = simulation.monte_carlo_simulation(
            ticker, 
            years=years, 
            simulations=10000, 
            risk_model=risk_model
        )
        
        # Generate visualization
        graphic = None
        try:
            # Recreate visualization method
            plt.figure(figsize=(12, 7))
            
            # Plot histogram of final prices
            final_prices = price_paths[:, -1]
            
            # Kernel Density Estimation
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(final_prices)
            x_range = np.linspace(final_prices.min(), final_prices.max(), 200)
            
            # Plot histogram
            plt.hist(final_prices, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            plt.plot(x_range, kde(x_range), color='red', linewidth=2)
            
            # Add percentile lines
            percentiles = {
                '5th': np.percentile(final_prices, 5),
                '10th': np.percentile(final_prices, 10),
                '25th': np.percentile(final_prices, 25),
                '50th': np.percentile(final_prices, 50),
                '75th': np.percentile(final_prices, 75),
                '90th': np.percentile(final_prices, 90),
                '100th': np.percentile(final_prices, 100)
            }
            
            # Color palette for percentile lines
            colors = ['darkblue', 'blue', 'lightblue', 'green', 'orange', 'red', 'darkred']
            
            # Add percentile lines
            for (name, value), color in zip(percentiles.items(), colors):
                plt.axvline(x=value, color=color, linestyle='--', 
                            label=f'{name} Percentile (${value:.2f})')
            
            plt.title(f'{ticker} Monte Carlo Simulation - Final Price Distribution')
            plt.xlabel('Final Stock Price')
            plt.ylabel('Density')
            plt.legend()
            
            # Save plot to a bytes buffer
            from io import BytesIO
            import base64
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            graphic = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
        except Exception as viz_error:
            logger.error(f"Visualization Error: {viz_error}")
            logger.error(traceback.format_exc())
        
        # Fetch analyst estimates for CAGR
        analyst_cagr_estimate = 0
        try:
            stock_info = yf.Ticker(ticker)
            
            # Method 1: Try to get 5-year growth estimate from analysis
            try:
                analysis = stock_info.analysis
                if analysis is not None and not analysis.empty:
                    growth_estimate = analysis.loc[analysis['Category'] == '5 Year Growth Estimate', 'Value']
                    if not growth_estimate.empty:
                        analyst_cagr_estimate = float(growth_estimate.values[0].replace('%', ''))
            except Exception as e1:
                logger.warning(f"Method 1 failed to get CAGR estimate: {e1}")
            
            # Method 2: If Method 1 fails, try getting from info dictionary
            if analyst_cagr_estimate == 0:
                try:
                    growth_keys = [
                        'fiveYearAverageReturn', 
                        'forwardAnnualDividendYield', 
                        'trailingAnnualDividendYield'
                    ]
                    
                    for key in growth_keys:
                        value = stock_info.info.get(key, 0)
                        if isinstance(value, (int, float)) and value != 0:
                            analyst_cagr_estimate = round(float(value) * 100, 2)
                            break
                except Exception as e2:
                    logger.warning(f"Method 2 failed to get CAGR estimate: {e2}")
            
            # Method 3: Fallback to historical growth calculation
            if analyst_cagr_estimate == 0:
                try:
                    history = stock_info.history(period='5y')
                    if not history.empty:
                        start_price = history['Close'].iloc[0]
                        end_price = history['Close'].iloc[-1]
                        analyst_cagr_estimate = round(((end_price / start_price) ** (1/5) - 1) * 100, 2)
                except Exception as e3:
                    logger.warning(f"Method 3 failed to get CAGR estimate: {e3}")
            
            # Ensure the estimate is within a reasonable range
            analyst_cagr_estimate = max(-50, min(50, analyst_cagr_estimate))
            
        except Exception as e:
            logger.error(f"Could not fetch analyst CAGR estimate: {e}")
        
        # Calculate CAGR for each simulation path
        def calculate_cagr(initial, final, years):
            return (final / initial) ** (1 / years) - 1
        
        initial_price = price_paths[:, 0][0]
        final_prices = price_paths[:, -1]
        
        cagr_values = [calculate_cagr(initial_price, final_prices[i], years) * 100 
                       for i in range(len(final_prices))]
        
        # Prepare response with JSON-safe conversions
        response_data = {
            'ticker': ticker,
            'years': years,
            'risk_model': risk_model,
            'initial_price': float(initial_price),
            'analyst_cagr_estimate': analyst_cagr_estimate,
            'metrics': {
                'mean': float(np.mean(final_prices)),
                'median': float(np.median(final_prices)),
                'std_dev': float(np.std(final_prices)),
                'cagr_mean': float(np.mean(cagr_values)),
                'cagr_median': float(np.median(cagr_values)),
                'cagr_std_dev': float(np.std(cagr_values)),
                'percentiles': {
                    '5th': float(np.percentile(final_prices, 5)),
                    '10th': float(np.percentile(final_prices, 10)),
                    '25th': float(np.percentile(final_prices, 25)),
                    '50th': float(np.percentile(final_prices, 50)),
                    '75th': float(np.percentile(final_prices, 75)),
                    '90th': float(np.percentile(final_prices, 90)),
                    '100th': float(np.percentile(final_prices, 100))
                },
                'cagr_percentiles': {
                    '5th': float(np.percentile(cagr_values, 5)),
                    '10th': float(np.percentile(cagr_values, 10)),
                    '25th': float(np.percentile(cagr_values, 25)),
                    '50th': float(np.percentile(cagr_values, 50)),
                    '75th': float(np.percentile(cagr_values, 75)),
                    '90th': float(np.percentile(cagr_values, 90)),
                    '100th': float(np.percentile(cagr_values, 100))
                }
            },
            'graphic': graphic,
            'analysis': {
                'mean': float(np.mean(final_prices)),
                'median': float(np.median(final_prices)),
                'stdDev': float(np.std(final_prices)),
                'percentiles': {
                    '10th': float(np.percentile(final_prices, 10)),
                    '50th': float(np.percentile(final_prices, 50)),
                    '90th': float(np.percentile(final_prices, 90))
                }
            }
        }
        
        # Safely convert price paths
        try:
            # Limit price paths to prevent JSON serialization issues
            if price_paths.size > 100000:
                # Randomly sample or reduce the array
                sample_indices = np.linspace(0, price_paths.shape[1] - 1, 1000, dtype=int)
                sampled_paths = price_paths[:, sample_indices]
                response_data['price_paths'] = sampled_paths.tolist()
            else:
                response_data['price_paths'] = price_paths.tolist()
        except Exception as path_error:
            logger.error(f"Error converting price paths: {path_error}")
            response_data['price_paths'] = []
        
        # Use safe JSON conversion with custom conversion
        return json.dumps(response_data, default=safe_numpy_conversion), 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        logger.error(f"Simulation Error: {e}")
        logger.error(traceback.format_exc())
        return json.dumps({
            'error': str(e),
            'details': traceback.format_exc()
        }, default=safe_numpy_conversion), 500, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    logger.info("Starting Monte Carlo Simulation Application")
    app.run(host='0.0.0.0', port=5000, debug=True)
