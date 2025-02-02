import os
import io
import sys
import base64
import logging
import traceback
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Ensure logging directory exists
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'monte_carlo_debug.log')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),  # Write mode, create in logs directory
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Startup logging
logger.info("=" * 50)
logger.info("Monte Carlo Stock Simulation Application")
logger.info("=" * 50)
logger.info(f"Log file: {log_file_path}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

# Log package versions
try:
    import pkg_resources
    packages = ['numpy', 'pandas', 'yfinance', 'flask', 'scipy', 'matplotlib']
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            logger.info(f"{package} version: {version}")
        except pkg_resources.DistributionNotFound:
            logger.warning(f"{package} not installed")
except ImportError:
    logger.warning("Could not log package versions")

# Debug function to log system information
def log_system_info():
    """Log system and environment information for debugging."""
    try:
        import platform
        
        logger.info("System Information:")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python Executable: {sys.executable}")
        
        # Log installed package versions
        logger.info("\nInstalled Packages:")
        try:
            import pkg_resources
            for package in ['yfinance', 'numpy', 'pandas', 'matplotlib', 'scipy', 'flask']:
                try:
                    version = pkg_resources.get_distribution(package).version
                    logger.info(f"{package}: {version}")
                except pkg_resources.DistributionNotFound:
                    logger.warning(f"{package} not installed")
        except ImportError:
            logger.warning("Could not log package versions")
        
        # Log environment variables
        logger.info("\nEnvironment Variables:")
        for key, value in os.environ.items():
            if key.lower() in ['path', 'python', 'home', 'user', 'temp']:
                logger.info(f"{key}: {value}")
    
    except Exception as e:
        logger.error(f"Error logging system information: {e}")

def validate_ticker(ticker):
    """
    Comprehensive stock ticker validation with detailed logging.
    
    Args:
        ticker (str): Stock ticker symbol to validate
    
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    # Check if input is a string
    if not isinstance(ticker, str):
        logger.warning(f"Invalid ticker type: {type(ticker)}")
        return False
    
    # Normalize ticker (uppercase, strip whitespace)
    ticker = ticker.strip().upper()
    
    # Comprehensive validation rules
    if not ticker:
        logger.warning("Empty ticker symbol")
        return False
    
    # Length check (1-7 characters)
    if len(ticker) < 1 or len(ticker) > 7:
        logger.warning(f"Ticker length invalid: {len(ticker)} characters")
        return False
    
    # Regex validation for allowed characters
    import re
    if not re.match(r'^[A-Z0-9.-]{1,7}$', ticker):
        logger.warning(f"Ticker contains invalid characters: {ticker}")
        return False
    
    # Optional: Additional validation using yfinance
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        
        # Check if stock info can be retrieved
        if not stock_info or 'symbol' not in stock_info:
            logger.warning(f"No stock information found for ticker: {ticker}")
            return False
        
        logger.info(f"Ticker {ticker} validated successfully")
        return True
    
    except Exception as e:
        logger.warning(f"Yfinance validation failed for {ticker}: {e}")
        return False

def fetch_stock_data(ticker, num_years=10, max_retries=3):
    """
    Advanced stock data retrieval with comprehensive error handling and diagnostics.
    
    Args:
        ticker (str): Stock ticker symbol
        num_years (int): Historical data years
        max_retries (int): Maximum retry attempts
    
    Returns:
        pd.DataFrame: Historical stock price data
    
    Raises:
        ValueError: For invalid input or insufficient data
        RuntimeError: For persistent retrieval failures
    """
    logger.info(f"Starting comprehensive stock data retrieval for {ticker}")
    
    # Validate ticker with comprehensive checks
    if not validate_ticker(ticker):
        logger.error(f"Invalid ticker format: {ticker}")
        raise ValueError(f"Invalid ticker format: {ticker}. Must be 1-7 alphanumeric characters.")
    
    # Extensive network and data retrieval diagnostics
    try:
        import socket
        import requests
        
        # Check internet connectivity
        try:
            socket.create_connection(("www.google.com", 80), timeout=5)
            logger.info("Internet connectivity verified")
        except (socket.error, socket.timeout):
            logger.warning("No internet connection detected")
            raise RuntimeError("No internet connection. Please check your network.")
        
        # Verify yfinance API accessibility
        try:
            response = requests.get("https://finance.yahoo.com", timeout=5)
            if response.status_code != 200:
                logger.warning(f"Yahoo Finance API not accessible. Status code: {response.status_code}")
        except requests.RequestException as req_error:
            logger.error(f"Failed to access Yahoo Finance: {req_error}")
    except ImportError:
        logger.warning("Could not perform additional network diagnostics")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} to fetch data for {ticker}")
            
            # Comprehensive date range setup
            end_date = datetime.now()
            start_date = end_date - timedelta(days=num_years*365)
            
            # Advanced stock data retrieval
            stock = yf.Ticker(ticker)
            
            # Verbose stock info logging with error handling
            try:
                stock_info = stock.info
                logger.debug(f"Stock info for {ticker}: {stock_info}")
                
                # Additional validation of stock info
                if not stock_info:
                    logger.warning(f"No stock information found for {ticker}")
            except Exception as info_error:
                logger.error(f"Error retrieving stock info for {ticker}: {info_error}")
            
            # Fetch historical data with extended timeout and error handling
            hist_data = stock.history(
                start=start_date, 
                end=end_date, 
                interval='1d', 
                timeout=15  # Extended timeout
            )
            
            # Comprehensive data validation
            if hist_data is None or hist_data.empty:
                logger.warning(f"No historical data found for {ticker}")
                
                # Additional diagnostic information
                try:
                    # Check if ticker exists in yfinance
                    ticker_exists = yf.Ticker(ticker).info is not None
                    logger.info(f"Ticker {ticker} exists in yfinance: {ticker_exists}")
                except Exception:
                    logger.warning(f"Could not verify ticker {ticker} existence")
                
                raise ValueError(f"Insufficient or no historical data for {ticker}")
            
            # Log data characteristics
            logger.info(f"Successfully retrieved {len(hist_data)} data points for {ticker}")
            logger.debug(f"Data range: {hist_data.index[0]} to {hist_data.index[-1]}")
            
            return hist_data
        
        except Exception as e:
            logger.error(f"Data retrieval error for {ticker} (Attempt {attempt + 1}): {e}")
            logger.error(traceback.format_exc())
            
            # Specific error handling
            if "Max retries exceeded" in str(e):
                logger.critical("Maximum network retries exceeded")
            elif "HTTP Error 404" in str(e):
                logger.error(f"Ticker {ticker} not found. Check if it's a valid stock symbol.")
            
            if attempt == max_retries - 1:
                logger.critical(f"Persistent failure retrieving data for {ticker}")
                raise RuntimeError(f"Could not retrieve stock data for {ticker}. Error: {str(e)}")
    
    # Fallback error (should never reach here)
    logger.critical("Unexpected termination of stock data retrieval")
    raise RuntimeError("Unhandled error in stock data retrieval")

def calculate_log_returns(prices):
    """
    Calculate log returns for more stable statistical analysis.
    
    Args:
        prices (pd.Series): Historical stock prices
    
    Returns:
        np.array: Log returns
    """
    return np.log(prices / prices.shift(1)).dropna().values

def estimate_parameters(log_returns):
    """
    Estimate drift and volatility parameters from log returns.
    
    Args:
        log_returns (pd.Series): Log returns of stock prices
    
    Returns:
        tuple: Estimated drift and volatility parameters
    """
    try:
        # Calculate basic statistics
        mu = np.mean(log_returns)
        sigma = np.std(log_returns)
        
        # Adjust for potential bias
        mu_adjusted = mu - 0.5 * sigma**2
        
        # Use ASCII-safe logging
        logger.info(f"Estimated Parameters: Drift = {mu_adjusted:.4f}, Volatility = {sigma:.4f}")
        
        return mu_adjusted, sigma
    
    except Exception as e:
        logger.error(f"Error estimating parameters: {e}")
        logger.error(traceback.format_exc())
        raise

def geometric_brownian_motion(S0, mu, sigma, T, dt, num_simulations):
    """
    Simulate stock prices using Geometric Brownian Motion.
    
    Args:
        S0 (float): Initial stock price
        mu (float): Drift (adjusted log return)
        sigma (float): Volatility
        T (int): Total trading days
        dt (float): Time step
        num_simulations (int): Number of simulation paths
    
    Returns:
        np.array: Simulated stock prices
    """
    # Generate random walks
    random_walks = np.random.standard_normal((num_simulations, T))
    
    # Initialize simulation array
    simulations = np.zeros((num_simulations, T))
    simulations[:, 0] = S0
    
    # Simulate paths
    for t in range(1, T):
        simulations[:, t] = simulations[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_walks[:, t]
        )
    
    return simulations

def run_monte_carlo_simulation(ticker, num_simulations=5000, num_years=10):
    """
    Perform advanced Monte Carlo simulation for stock price projection.
    
    Args:
        ticker (str): Stock ticker symbol
        num_simulations (int): Number of simulation runs
        num_years (int): Number of years to simulate
    
    Returns:
        dict: Comprehensive simulation results with plot, percentiles, and statistical insights
    
    Raises:
        ValueError: For invalid ticker or insufficient data
        RuntimeError: For simulation failures
    """
    try:
        # Validate inputs
        if not validate_ticker(ticker):
            logger.error(f"Invalid ticker symbol: {ticker}")
            raise ValueError(f"Invalid ticker symbol: {ticker}. Must be 1-7 alphanumeric characters.")
        
        if num_simulations < 1000 or num_simulations > 10000:
            logger.warning(f"Adjusting num_simulations to recommended range: {num_simulations}")
            num_simulations = max(1000, min(num_simulations, 10000))
        
        if num_years < 1 or num_years > 20:
            logger.warning(f"Adjusting num_years to recommended range: {num_years}")
            num_years = max(1, min(num_years, 20))
        
        # Fetch historical data
        try:
            hist_data = fetch_stock_data(ticker, num_years)
        except Exception as fetch_error:
            logger.error(f"Data fetch failed for {ticker}: {fetch_error}")
            raise RuntimeError(f"Could not retrieve historical data for {ticker}. {str(fetch_error)}")
        
        # Calculate log returns
        log_returns = calculate_log_returns(hist_data['Close'])
        
        # Estimate parameters
        current_price = hist_data['Close'].iloc[-1]
        mu, sigma = estimate_parameters(log_returns)
        
        # Simulation parameters
        trading_days = 252 * num_years
        dt = 1/trading_days
        
        # Run simulation
        simulations = geometric_brownian_motion(
            S0=current_price, 
            mu=mu, 
            sigma=sigma, 
            T=trading_days, 
            dt=dt, 
            num_simulations=num_simulations
        )
        
        # Calculate percentiles and statistical insights
        final_prices = simulations[:, -1]
        percentiles = {p: np.percentile(final_prices, p) for p in [10, 25, 50, 75, 90]}
        
        # Advanced statistical analysis
        mean_final_price = np.mean(final_prices)
        std_final_price = np.std(final_prices)
        skewness = stats.skew(final_prices)
        kurtosis = stats.kurtosis(final_prices)
        
        # Confidence intervals
        confidence_intervals = {
            p: stats.t.interval(
                alpha=0.95, 
                df=len(final_prices)-1, 
                loc=np.mean(final_prices[final_prices <= percentiles[p]]), 
                scale=stats.sem(final_prices[final_prices <= percentiles[p]])
            ) for p in [10, 25, 50, 75, 90]
        }
        
        # Create visualization with enhanced aesthetics
        plt.figure(figsize=(15, 9), facecolor='#121212')
        plt.style.use('dark_background')
        plt.title(f'{ticker} Monte Carlo Simulation', color='white', fontsize=18, fontweight='bold')
        plt.xlabel('Trading Days', color='white', fontsize=12)
        plt.ylabel('Stock Price ($)', color='white', fontsize=12)
        
        # Plot first 100 simulation paths with gradient
        for i in range(100):
            plt.plot(simulations[i], linewidth=0.7, alpha=0.2, color=plt.cm.plasma(i/100))
        
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#121212', edgecolor='none', dpi=300)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        logger.info(f"Monte Carlo simulation completed successfully for {ticker}")
        
        return {
            'current_price': current_price,
            'percentiles': percentiles,
            'confidence_intervals': {str(k): list(v) for k, v in confidence_intervals.items()},
            'statistical_insights': {
                'mean_final_price': mean_final_price,
                'std_final_price': std_final_price,
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'plot': plot_base64,
            'simulation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'drift': mu,
                'volatility': sigma,
                'trading_days': trading_days,
                'num_simulations': num_simulations
            }
        }
    
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed for {ticker}: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Simulation failed: {e}")

# Create Flask application
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))
CORS(app)

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Advanced stock simulation route with comprehensive error handling and logging.
    """
    # Extensive logging for each simulation request
    logger.info("Received simulation request")
    
    try:
        # Parse request data with error handling
        try:
            data = request.get_json()
            logger.debug(f"Received request data: {data}")
        except Exception as parse_error:
            logger.error(f"Failed to parse request JSON: {parse_error}")
            return jsonify({
                'error': 'Invalid request',
                'details': 'Could not parse request data. Ensure JSON format.'
            }), 400
        
        # Validate request data
        if not isinstance(data, dict):
            logger.error(f"Invalid request data type: {type(data)}")
            return jsonify({
                'error': 'Invalid request format',
                'details': 'Request must be a JSON object'
            }), 400
        
        # Extract and validate ticker
        ticker = str(data.get('ticker', '')).strip().upper()
        logger.info(f"Requested ticker: {ticker}")
        
        # Comprehensive ticker validation
        if not ticker:
            logger.error("Empty ticker symbol received")
            return jsonify({
                'error': 'Ticker symbol cannot be empty',
                'details': 'Please provide a valid stock ticker (1-7 alphanumeric characters)'
            }), 400
        
        # Validate ticker using existing validation function
        if not validate_ticker(ticker):
            logger.error(f"Invalid ticker format: {ticker}")
            return jsonify({
                'error': 'Invalid ticker format',
                'details': 'Ticker must be 1-7 alphanumeric characters (A-Z, 0-9, ., -)'
            }), 400
        
        # Extract and validate simulation parameters
        try:
            num_simulations = max(1000, min(int(data.get('num_simulations', 5000)), 10000))
            num_years = max(1, min(int(data.get('num_years', 10)), 20))
            
            logger.info(f"Simulation parameters: {num_simulations} simulations, {num_years} years")
        except ValueError as ve:
            logger.error(f"Invalid simulation parameters: {ve}")
            return jsonify({
                'error': 'Invalid simulation parameters',
                'details': 'Number of simulations and years must be integers'
            }), 400
        
        # Comprehensive simulation execution with detailed error handling
        try:
            result = run_monte_carlo_simulation(ticker, num_simulations, num_years)
            logger.info(f"Simulation completed successfully for {ticker}")
            return jsonify(result), 200
        
        except ValueError as ve:
            logger.error(f"Validation error in simulation for {ticker}: {ve}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Simulation validation failed',
                'details': str(ve)
            }), 400
        
        except RuntimeError as re:
            logger.error(f"Runtime error in simulation for {ticker}: {re}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Simulation failed',
                'details': str(re)
            }), 500
    
    except Exception as e:
        logger.critical(f"Unexpected critical error in simulation request: {e}")
        logger.critical(traceback.format_exc())
        return jsonify({
            'error': 'Critical simulation error',
            'details': f'An unexpected error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Determine port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Ensure templates directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'templates'), exist_ok=True)
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)

# Call system info logging at module import
log_system_info()
