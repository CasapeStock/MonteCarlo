# Monte Carlo Stock Simulation App

## Overview
A comprehensive application for performing Monte Carlo Simulations on stock tickers, providing probabilistic analysis of stock performance over 10 years.

## Repository
- **Name:** MonteCarlo
- **Owner:** CasapeStock
- **URL:** https://github.com/CasapeStock/MonteCarlo

## Features
- Input any stock ticker
- Generate Monte Carlo Simulation graph
- Display percentile-based stock price predictions
- Dark, professional UI design
- Localhost web server for easy access
- Cross-platform compatibility (Web, Mobile, Desktop)

## Technologies
- Python
- Kivy
- Flask
- yfinance
- NumPy
- Pandas
- Matplotlib

## Prerequisites
- Python 3.8+
- Git
- GitHub Account

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CasapeStock/MonteCarlo.git
   cd MonteCarlo
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Web Server
```bash
python app/server.py
```
- Access at `http://localhost:5000`

### Mobile App
```bash
python app/main.py
```

## Deployment Options
- Local development
- Web server (Flask)
- Mobile app (Kivy)
- Android packaging with Buildozer

## Android Packaging
```bash
buildozer android debug deploy run
```

## Contribution
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Disclaimer
Stock simulations are for educational purposes only. Not financial advice.

## License
[Specify your license here]
