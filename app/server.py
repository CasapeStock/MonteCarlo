from flask import request, render_template, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

from . import create_app

app = create_app()

def monte_carlo_simulation(ticker, years=10, simulations=1000, risk_model='moderate'):
    # Input validation
    if not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("Invalid ticker symbol. Must be a non-empty string.")
    
    if not isinstance(years, (int, float)) or years <= 0:
        raise ValueError("Years must be a positive number.")
    
    if not isinstance(simulations, int) or simulations <= 0:
        raise ValueError("Number of simulations must be a positive integer.")
    
    # Validate risk model
    valid_risk_models = ['conservative', 'moderate', 'aggressive']
    if risk_model.lower() not in valid_risk_models:
        raise ValueError(f"Invalid risk model. Must be one of {valid_risk_models}")

    try:
        # Fetch stock data with extended error handling
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=f"{max(10, years)}y")
        
        # Check if historical data is available
        if hist_data.empty:
            raise ValueError(f"No historical data found for ticker: {ticker}")
        
        # More comprehensive return calculations with error handling
        close_prices = hist_data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Validate return calculations
        if len(log_returns) == 0:
            raise ValueError("Insufficient data to calculate returns")
        
        # Advanced parameters with additional checks
        mu = log_returns.mean()  # Mean of log returns
        sigma = log_returns.std()  # Standard deviation of log returns
        
        # Validate statistical parameters
        if np.isnan(mu) or np.isnan(sigma):
            raise ValueError("Unable to calculate valid statistical parameters")
        
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
        
        # Adjusted volatility and return with additional safety checks
        adjusted_sigma = sigma * risk_params['volatility_multiplier']
        adjusted_mu = mu + risk_params['return_adjustment'] * sigma
        
        # Last known stock price with validation
        last_price = hist_data['Close'][-1]
        if last_price <= 0:
            raise ValueError(f"Invalid last price for {ticker}: {last_price}")
        
        # Monte Carlo simulation with more realistic stochastic process
        trading_days = years * 252  # Typical trading days in a year
        simulations_data = np.zeros((simulations, trading_days))
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        for i in range(simulations):
            # Geometric Brownian Motion for more realistic stock price simulation
            daily_returns = np.random.normal(
                (adjusted_mu - 0.5 * adjusted_sigma**2), 
                adjusted_sigma, 
                trading_days
            )
            
            # Cumulative returns and price series with additional checks
            cumulative_returns = np.cumsum(daily_returns)
            price_series = last_price * np.exp(cumulative_returns)
            
            # Validate price series
            if np.any(np.isinf(price_series)) or np.any(np.isnan(price_series)):
                raise ValueError(f"Invalid price series generated for simulation {i}")
            
            simulations_data[i] = price_series
        
        return simulations_data
    
    except Exception as e:
        # Comprehensive error logging
        print(f"Monte Carlo Simulation Error for {ticker}: {str(e)}")
        raise

def plot_monte_carlo(simulations_data, ticker):
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

def get_stock_details(ticker):
    stock = yf.Ticker(ticker)
    
    # Current price
    current_price = stock.history(period='1d')['Close'][-1]
    
    # 5-year historical data for CAGR calculation
    hist_data = stock.history(period='5y')
    start_price = hist_data['Close'].iloc[0]
    end_price = hist_data['Close'].iloc[-1]
    
    # Calculate 5-year CAGR
    years = 5
    cagr = ((end_price / start_price) ** (1/years) - 1) * 100
    
    # Additional stock info
    info = stock.info
    
    return {
        'current_price': round(current_price, 2),
        'cagr_5y': round(cagr, 2),
        'company_name': info.get('longName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock_details', methods=['POST'])
def stock_details():
    ticker = request.form.get('ticker', 'AAPL')
    try:
        details = get_stock_details(ticker)
        return jsonify(details)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/simulate', methods=['POST'])
def simulate():
    ticker = request.form.get('ticker', 'AAPL')
    risk_model = request.form.get('risk_model', 'moderate')
    
    try:
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
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
