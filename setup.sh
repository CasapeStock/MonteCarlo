#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Stock Evaluation App setup"

# Optional: Add remote repository (replace with your repository URL)
# git remote add origin https://github.com/yourusername/StockControl.git

echo "Setup complete! Virtual environment created and git repository initialized."
