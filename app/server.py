from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def run_monte_carlo_simulation(ticker, num_simulations=1000, num_years=10):
    try:
        # Fetch stock data
        stock_data = yf.Ticker(ticker)
        historical_prices = stock_data.history(period='5y')['Close']
        
        # Calculate returns
        returns = historical_prices.pct_change()
        last_price = historical_prices.iloc[-1]
        
        # Simulation parameters
        num_days = 252 * num_years  # Trading days in specified years
        
        # Run simulation
        simulations = np.zeros((num_simulations, num_days))
        for i in range(num_simulations):
            daily_returns = np.random.normal(
                returns.mean(), 
                returns.std(), 
                num_days
            )
            simulations[i] = last_price * (1 + daily_returns).cumprod()
        
        # Calculate percentiles
        percentiles = [0, 5, 10, 20, 40, 60, 80, 100]
        final_prices = simulations[:, -1]
        price_percentiles = np.percentile(final_prices, percentiles)
        
        # Create plot
        plt.figure(figsize=(10, 6), facecolor='#1a1a1a')
        plt.style.use('dark_background')
        plt.plot(simulations.T, color='lightblue', alpha=0.1)
        plt.title(f'{ticker} Monte Carlo Simulation', color='white')
        plt.xlabel('Trading Days', color='white')
        plt.ylabel('Stock Price', color='white')
        plt.tick_params(colors='white')
        
        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor='#1a1a1a')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'percentiles': dict(zip(percentiles, price_percentiles)),
            'plot': plot_data
        }
    
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    ticker = request.form.get('ticker', '').upper()
    result = run_monte_carlo_simulation(ticker)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
