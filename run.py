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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app():
    app = Flask(__name__, 
                template_folder=os.path.join(BASE_DIR, 'app', 'templates'),
                static_folder=os.path.join(BASE_DIR, 'app', 'static'))
    
    # Add error handling
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled Exception: {e}", exc_info=True)
        return f"An error occurred: {str(e)}", 500

    @app.route('/')
    def index():
        logger.debug("Index route accessed")
        return render_template('index.html')

    @app.route('/simulate', methods=['POST'])
    def simulate():
        try:
            ticker = request.form.get('ticker', 'AAPL')
            risk_model = request.form.get('risk_model', 'moderate')
            
            logger.debug(f"Simulation request: Ticker={ticker}, Risk Model={risk_model}")
            
            simulations_data = monte_carlo_simulation(ticker, risk_model=risk_model)
            graphic = plot_monte_carlo(simulations_data, ticker)
            
            return jsonify({
                'graphic': graphic,
                'percentiles': {
                    '10th': np.percentile(simulations_data, 10, axis=0)[-1],
                    '50th': np.percentile(simulations_data, 50, axis=0)[-1],
                    '90th': np.percentile(simulations_data, 90, axis=0)[-1]
                }
            })
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 400

    return app

def monte_carlo_simulation(ticker, years=10, simulations=1000, risk_model='moderate'):
    try:
        stock = yf.Ticker(ticker)
        
        # Extended historical data for more accurate analysis
        hist_data = stock.history(period=f"{max(10, years)}y")
        
        # More comprehensive return calculations
        close_prices = hist_data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Advanced parameters
        mu = log_returns.mean()  # Mean of log returns
        sigma = log_returns.std()  # Standard deviation of log returns
        
        # Risk model adjustments with more nuanced approach
        risk_factors = {
            'conservative': {
                'volatility_multiplier': 0.7,
                'return_adjustment': -0.5  # Reduce expected returns
            },
            'moderate': {
                'volatility_multiplier': 1.0,
                'return_adjustment': 0
            },
            'aggressive': {
                'volatility_multiplier': 1.5,
                'return_adjustment': 0.5  # Increase potential returns
            }
        }
        
        # Get current risk model parameters
        risk_params = risk_factors.get(risk_model.lower(), risk_factors['moderate'])
        
        # Adjusted volatility and return
        adjusted_sigma = sigma * risk_params['volatility_multiplier']
        adjusted_mu = mu + risk_params['return_adjustment'] * sigma
        
        # Last known stock price
        last_price = hist_data['Close'][-1]
        
        # Monte Carlo simulation with more realistic stochastic process
        trading_days = years * 252  # Typical trading days in a year
        simulations_data = np.zeros((simulations, trading_days))
        
        for i in range(simulations):
            # Geometric Brownian Motion for more realistic stock price simulation
            daily_returns = np.random.normal(
                (adjusted_mu - 0.5 * adjusted_sigma**2), 
                adjusted_sigma, 
                trading_days
            )
            
            # Cumulative returns and price series
            cumulative_returns = np.cumsum(daily_returns)
            price_series = last_price * np.exp(cumulative_returns)
            
            simulations_data[i] = price_series
        
        return simulations_data
    except Exception as e:
        logger.error(f"Simulation calculation error: {e}", exc_info=True)
        raise

def plot_monte_carlo(simulations_data, ticker):
    try:
        plt.figure(figsize=(12, 7))
        plt.title(f'Monte Carlo Simulation: {ticker}', fontsize=15)
        plt.xlabel('Trading Days', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        
        # Plot individual simulation paths (semi-transparent)
        for simulation in simulations_data[:50]:  # Plot first 50 paths
            plt.plot(simulation, color='lightgray', alpha=0.1)
        
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
        buffer.close()
        plt.close()
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        return graphic
    except Exception as e:
        logger.error(f"Plot generation error: {e}", exc_info=True)
        raise

def main():
    app = create_app()
    logger.info("Starting Flask application")
    try:
        app.run(
            host='0.0.0.0',  # Listen on all available interfaces
            port=5000, 
            debug=True
        )
    except Exception as e:
        logger.critical(f"Failed to start Flask application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
