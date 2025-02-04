from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='app/templates', 
            static_folder='app/static')

def monte_carlo_simulation(ticker='AAPL', years=10, simulations=1000, risk_model='moderate'):
    """Simplified Monte Carlo simulation for local testing"""
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=f"{max(10, years)}y")
        
        if hist_data.empty:
            raise ValueError(f"No data for {ticker}")
        
        # Calculate log returns
        close_prices = hist_data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Risk model adjustments
        risk_factors = {
            'conservative': 0.7,
            'moderate': 1.0,
            'aggressive': 1.5
        }
        
        # Parameters
        last_price = close_prices.iloc[-1]
        mu = log_returns.mean()
        sigma = log_returns.std()
        
        # Adjust volatility
        adjusted_sigma = sigma * risk_factors.get(risk_model, 1.0)
        
        # Simulation
        trading_days = years * 252
        np.random.seed(42)
        
        simulations_data = np.zeros((simulations, trading_days))
        for i in range(simulations):
            daily_returns = np.random.normal(
                (mu - 0.5 * adjusted_sigma**2), 
                adjusted_sigma, 
                trading_days
            )
            cumulative_returns = np.cumsum(daily_returns)
            price_series = last_price * np.exp(cumulative_returns)
            simulations_data[i] = price_series
        
        return simulations_data
    
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise

def plot_monte_carlo(simulations_data, ticker):
    """Generate plot for Monte Carlo simulation"""
    plt.figure(figsize=(10, 6))
    plt.title(f'Monte Carlo Simulation: {ticker}')
    
    # Plot percentile lines
    percentiles = [10, 50, 90]
    colors = ['red', 'green', 'blue']
    
    for percentile, color in zip(percentiles, colors):
        plt.plot(
            np.percentile(simulations_data, percentile, axis=0), 
            color=color, 
            label=f'{percentile}th Percentile'
        )
    
    plt.xlabel('Trading Days')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    """Handle simulation request"""
    try:
        ticker = request.form.get('ticker', 'AAPL').upper()
        risk_model = request.form.get('risk_model', 'moderate').lower()
        
        logger.info(f"Simulation request: {ticker}, {risk_model}")
        
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
        logger.error(f"Simulation error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting local test server")
    app.run(
        host='127.0.0.1',  # Explicitly use localhost
        port=5000, 
        debug=True
    )
