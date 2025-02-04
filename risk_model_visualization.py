import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def simulate_risk_models(ticker='AAPL', years=10, simulations=1000):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period=f"{years}y")
    
    # Calculate daily returns
    returns = hist_data['Close'].pct_change()
    
    # Parameters for simulation
    last_price = hist_data['Close'][-1]
    daily_return = returns.mean()
    daily_volatility = returns.std()
    
    # Risk model adjustments
    risk_factors = {
        'Conservative': 0.5,
        'Moderate': 1.0,
        'Aggressive': 1.5
    }
    
    plt.figure(figsize=(15, 5))
    
    for i, (model, multiplier) in enumerate(risk_factors.items(), 1):
        # Adjust volatility based on risk model
        adjusted_volatility = daily_volatility * multiplier
        
        # Monte Carlo simulation
        simulations_data = np.zeros((simulations, years * 252))
        for j in range(simulations):
            daily_returns = np.random.normal(daily_return, adjusted_volatility, years * 252) + 1
            price_series = last_price * daily_returns.cumprod()
            simulations_data[j] = price_series
        
        plt.subplot(1, 3, i)
        percentiles = [10, 50, 90]
        colors = ['red', 'green', 'blue']
        
        for percentile, color in zip(percentiles, colors):
            plt.plot(np.percentile(simulations_data, percentile, axis=0), 
                     color=color, label=f'{percentile}th Percentile')
        
        plt.title(f'{model} Risk Model\nVolatility Multiplier: {multiplier}')
        plt.xlabel('Trading Days')
        plt.ylabel('Stock Price')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('risk_models_comparison.png')
    plt.close()

# Run the visualization
simulate_risk_models()
