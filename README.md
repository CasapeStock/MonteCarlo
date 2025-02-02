# Avaliando Ações (Stock Evaluator)

## Description
A mobile application for performing Monte Carlo Simulations on stock tickers, providing probabilistic analysis of stock performance over 10 years.

## Features
- Input any stock ticker
- Generate Monte Carlo Simulation graph
- Display percentile-based stock price predictions
- Dark, professional UI design
- Localhost web server for easy access
- Git version control

## Prerequisites
- Python 3.8+
- Git
- Kivy
- yfinance
- numpy
- pandas
- matplotlib
- Flask

## Repository Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StockControl.git
   cd StockControl
   ```

## Installation
1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

### Mobile App
```bash
python app/main.py
```

### Localhost Web Server
```bash
python app/server.py
```
- Access the web interface at `http://localhost:5000`

## Git Workflow
- Create a new branch for features:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Commit changes:
  ```bash
  git add .
  git commit -m "Description of changes"
  ```
- Push to remote:
  ```bash
  git push origin feature/your-feature-name
  ```

## Deployment Options
- Local development
- Web server (Flask)
- Mobile app (Kivy)

## Android Packaging
Use Buildozer to package for Android:
```bash
buildozer android debug deploy run
```

## Disclaimer
Stock simulations are for educational purposes only. Not financial advice.
