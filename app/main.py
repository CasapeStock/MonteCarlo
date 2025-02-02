import kivy
kivy.require('2.2.1')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd

class MonteCarloSimulationApp(App):
    def build(self):
        # Main layout
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        layout.canvas.before.clear()
        layout.canvas.before.add(Color(0.1, 0.1, 0.1, 1))  # Dark background
        
        # Ticker input
        ticker_layout = BoxLayout(size_hint_y=None, height=50)
        self.ticker_input = TextInput(
            multiline=False, 
            hint_text='Enter Stock Ticker (e.g., SOFI)', 
            size_hint_x=0.7
        )
        simulate_btn = Button(
            text='Simulate', 
            size_hint_x=0.3, 
            on_press=self.run_monte_carlo
        )
        ticker_layout.add_widget(self.ticker_input)
        ticker_layout.add_widget(simulate_btn)
        layout.add_widget(ticker_layout)
        
        # Results area
        self.results_layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.results_layout)
        
        return layout
    
    def run_monte_carlo(self, instance):
        ticker = self.ticker_input.text.upper()
        
        try:
            # Fetch stock data
            stock_data = yf.Ticker(ticker)
            historical_prices = stock_data.history(period='5y')['Close']
            
            # Monte Carlo Simulation
            returns = historical_prices.pct_change()
            last_price = historical_prices.iloc[-1]
            
            # Simulation parameters
            num_simulations = 1000
            num_days = 252 * 10  # 10 years of trading days
            
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
            
            # Clear previous results
            self.results_layout.clear_widgets()
            
            # Plot simulation
            plt.figure(figsize=(10, 6), facecolor='#1a1a1a')
            plt.style.use('dark_background')
            plt.plot(simulations.T, color='lightblue', alpha=0.1)
            plt.title(f'{ticker} Monte Carlo Simulation', color='white')
            plt.xlabel('Trading Days', color='white')
            plt.ylabel('Stock Price', color='white')
            plt.tick_params(colors='white')
            
            # Add plot to layout
            canvas = FigureCanvasKivyAgg(plt.gcf())
            self.results_layout.add_widget(canvas)
            
            # Create percentile table
            table_layout = BoxLayout(size_hint_y=None, height=200)
            table_text = "\n".join([
                f"{p}% Percentile: ${price:.2f}" 
                for p, price in zip(percentiles, price_percentiles)
            ])
            table_label = Label(
                text=table_text, 
                color=(1,1,1,1),  # White text
                font_size='14sp'
            )
            table_layout.add_widget(table_label)
            self.results_layout.add_widget(table_layout)
            
            plt.close()  # Close the matplotlib figure to free memory
            
        except Exception as e:
            error_label = Label(
                text=f"Error: {str(e)}", 
                color=(1,0,0,1)  # Red error text
            )
            self.results_layout.add_widget(error_label)

def main():
    MonteCarloSimulationApp().run()

if __name__ == '__main__':
    main()
