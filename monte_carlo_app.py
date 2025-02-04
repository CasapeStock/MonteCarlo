import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import base64
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('monte_carlo_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Determine base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app():
    """Create and configure Flask application"""
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'app', 'templates'),
        static_folder=os.path.join(BASE_DIR, 'app', 'static')
    )
    
    @app.route('/')
    def index():
        """Render main application page"""
        logger.info("Index page accessed")
        return render_template('index.html')
    
    @app.route('/simulate', methods=['POST'])
    def simulate():
        """Perform Monte Carlo simulation"""
        try:
            ticker = request.form.get('ticker', 'AAPL').upper()
            risk_model = request.form.get('risk_model', 'moderate').lower()
            
            logger.info(f"Simulation request: Ticker={ticker}, Risk Model={risk_model}")
            
            # Validate inputs
            valid_risk_models = ['conservative', 'moderate', 'aggressive']
            if risk_model not in valid_risk_models:
                raise ValueError(f"Invalid risk model. Must be one of {valid_risk_models}")
            
            # Perform simulation
            simulations_data = monte_carlo_simulation(ticker, risk_model=risk_model)
            graphic = plot_monte_carlo(simulations_data, ticker)
            
            return jsonify({
                'graphic': graphic,
                'percentiles': {
                    '10th': round(np.percentile(simulations_data, 10, axis=0)[-1], 2),
                    '50th': round(np.percentile(simulations_data, 50, axis=0)[-1], 2),
                    '90th': round(np.percentile(simulations_data, 90, axis=0)[-1], 2)
                }
            })
        
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 400

    return app

def monte_carlo_simulation(ticker, years=10, simulations=1000, risk_model='moderate'):
    """Perform Monte Carlo stock price simulation"""
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=f"{max(10, years)}y")
        
        if hist_data.empty:
            raise ValueError(f"No historical data found for ticker: {ticker}")
        
        # Calculate log returns
        close_prices = hist_data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Risk model parameters
        risk_factors = {
            'conservative': {'volatility': 0.7, 'return_adj': -0.5},
            'moderate': {'volatility': 1.0, 'return_adj': 0},
            'aggressive': {'volatility': 1.5, 'return_adj': 0.5}
        }
        
        # Get risk model parameters
        risk_params = risk_factors.get(risk_model, risk_factors['moderate'])
        
        # Calculate parameters
        mu = log_returns.mean()
        sigma = log_returns.std()
        
        # Adjust parameters based on risk model
        adjusted_sigma = sigma * risk_params['volatility']
        adjusted_mu = mu + risk_params['return_adj'] * sigma
        
        # Last stock price
        last_price = close_prices.iloc[-1]
        
        # Simulation
        trading_days = years * 252
        np.random.seed(42)  # For reproducibility
        
        simulations_data = np.zeros((simulations, trading_days))
        for i in range(simulations):
            daily_returns = np.random.normal(
                (adjusted_mu - 0.5 * adjusted_sigma**2), 
                adjusted_sigma, 
                trading_days
            )
            cumulative_returns = np.cumsum(daily_returns)
            price_series = last_price * np.exp(cumulative_returns)
            simulations_data[i] = price_series
        
        return simulations_data
    
    except Exception as e:
        logger.error(f"Simulation error for {ticker}: {e}")
        raise

def plot_monte_carlo(simulations_data, ticker):
    """Generate Monte Carlo simulation plot"""
    plt.figure(figsize=(12, 7))
    plt.title(f'Monte Carlo Simulation: {ticker}', fontsize=15)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    
    # Plot percentile lines
    percentiles = [10, 50, 90]
    colors = ['red', 'green', 'blue']
    labels = ['10th Percentile', 'Median', '90th Percentile']
    
    for percentile, color, label in zip(percentiles, colors, labels):
        plt.plot(
            np.percentile(simulations_data, percentile, axis=0), 
            color=color, 
            linewidth=2, 
            label=label
        )
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

def main():
    """Main application entry point"""
    app = create_app()
    logger.info("Starting Monte Carlo Simulation App")
    
    try:
        app.run(
            host='0.0.0.0',  # Listen on all interfaces
            port=5000, 
            debug=True
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
