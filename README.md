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
- GitHub repository integration

## Prerequisites
- Python 3.8+
- Git
- GitHub Account
- Kivy
- yfinance
- numpy
- pandas
- matplotlib
- Flask

## GitHub Repository Setup

### Option 1: Automatic Setup (Recommended)
1. Install GitHub CLI
   ```bash
   # For Windows, use Chocolatey or download from GitHub CLI website
   choco install gh
   ```

2. Login to GitHub CLI
   ```bash
   gh auth login
   ```

3. Run GitHub setup script
   ```bash
   chmod +x github_setup.sh
   ./github_setup.sh
   ```

### Option 2: Manual Setup
1. Create a new repository on GitHub
   - Go to [GitHub New Repository](https://github.com/new)
   - Name: StockControl
   - Description: Stock Evaluation Mobile and Web Application
   - Public visibility
   - Do NOT initialize with README, .gitignore, or license

2. In your local repository, set up remote:
   ```bash
   git remote add origin https://github.com/yourusername/StockControl.git
   git branch -M main
   git push -u origin main
   ```

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
