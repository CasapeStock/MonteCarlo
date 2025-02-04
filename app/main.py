from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import os
import json

class MonteCarloApp(App):
    def build(self):
        Window.clearcolor = get_color_from_hex('#121212')
        
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        title = Label(
            text='Monte Carlo Stock Simulation', 
            font_size='20sp', 
            color=get_color_from_hex('#FFFFFF')
        )
        layout.add_widget(title)
        
        self.ticker_input = TextInput(
            multiline=False, 
            hint_text='Enter Stock Ticker (e.g., AAPL)',
            background_color=get_color_from_hex('#1E1E1E'),
            foreground_color=get_color_from_hex('#FFFFFF')
        )
        layout.add_widget(self.ticker_input)
        
        # Risk model spinner
        risk_models = ['Moderate', 'Conservative', 'Aggressive']
        self.risk_model_spinner = Spinner(
            text='Risk Model',
            values=risk_models,
            background_color=get_color_from_hex('#007BFF'),
            color=get_color_from_hex('#FFFFFF')
        )
        layout.add_widget(self.risk_model_spinner)
        
        simulate_btn = Button(
            text='Simulate', 
            background_color=get_color_from_hex('#007BFF'),
            color=get_color_from_hex('#FFFFFF')
        )
        simulate_btn.bind(on_press=self.simulate)
        layout.add_widget(simulate_btn)

        # Add result display area
        self.result_label = Label(
            text='Simulation results will appear here',
            font_size='16sp',
            color=get_color_from_hex('#FFFFFF')
        )
        layout.add_widget(self.result_label)

        return layout

    def monte_carlo_simulation(self, ticker, years=10, simulations=1000, risk_model='Moderate'):
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=f"{years}y")
        
        # Calculate daily returns
        returns = hist_data['Close'].pct_change()
        
        # Parameters for simulation based on risk model
        last_price = hist_data['Close'][-1]
        daily_return = returns.mean()
        daily_volatility = returns.std()
        
        # Risk model adjustments
        risk_factors = {
            'Conservative': 0.5,   # Lower volatility, lower returns
            'Moderate': 1.0,       # Standard market behavior
            'Aggressive': 1.5      # Higher volatility, potential for higher returns
        }
        
        risk_multiplier = risk_factors.get(risk_model, 1.0)
        
        # Adjust volatility based on risk model
        adjusted_volatility = daily_volatility * risk_multiplier
        
        # Monte Carlo simulation
        simulations_data = np.zeros((simulations, years * 252))
        for i in range(simulations):
            daily_returns = np.random.normal(daily_return, adjusted_volatility, years * 252) + 1
            price_series = last_price * daily_returns.cumprod()
            simulations_data[i] = price_series
        
        return simulations_data
    
    def plot_monte_carlo(self, simulations_data, ticker):
        plt.figure(figsize=(10, 6))
        plt.title(f'Monte Carlo Simulation: {ticker}')
        plt.xlabel('Trading Days')
        plt.ylabel('Stock Price')
        
        percentiles = [10, 50, 90]
        colors = ['red', 'green', 'blue']
        
        for percentile, color in zip(percentiles, colors):
            plt.plot(np.percentile(simulations_data, percentile, axis=0), 
                     color=color, label=f'{percentile}th Percentile')
        
        plt.legend()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def get_stock_details(self, ticker):
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
    
    def simulate(self, instance):
        ticker = self.ticker_input.text.upper()
        risk_model = self.risk_model_spinner.text
        
        try:
            # Get stock details
            stock_details = self.get_stock_details(ticker)
            
            # Update stock details label
            details_text = (
                f"Company: {stock_details['company_name']}\n"
                f"Current Price: ${stock_details['current_price']}\n"
                f"5-Year CAGR: {stock_details['cagr_5y']}%\n"
                f"Sector: {stock_details['sector']}\n"
                f"Market Cap: ${stock_details['market_cap']}"
            )
            self.result_label.text = details_text
            
            # Perform Monte Carlo simulation
            simulations_data = self.monte_carlo_simulation(ticker, risk_model=risk_model)
            graphic = self.plot_monte_carlo(simulations_data, ticker)
            
            # Update image
            # self.result_image.source = f'data:image/png;base64,{graphic}'
            
            # Update percentiles
            percentiles = {
                '10th': np.percentile(simulations_data, 10, axis=0)[-1],
                '50th': np.percentile(simulations_data, 50, axis=0)[-1],
                '90th': np.percentile(simulations_data, 90, axis=0)[-1]
            }
            
            percentile_text = (
                f"10th Percentile: ${percentiles['10th']:.2f}\n"
                f"50th Percentile: ${percentiles['50th']:.2f}\n"
                f"90th Percentile: ${percentiles['90th']:.2f}\n"
                f"Risk Model: {risk_model}"
            )
            
            self.result_label.text += '\n\n' + percentile_text
        
        except Exception as e:
            self.result_label.text = f'Error: {str(e)}'

if __name__ == '__main__':
    MonteCarloApp().run()
