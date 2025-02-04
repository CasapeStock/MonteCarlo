#!/bin/bash

# Set Git user configuration
git config --global user.name "CasapeStock"
git config --global user.email "manaia@123"

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Monte Carlo Stock Simulation App"

# Push to GitHub
git push -u origin main

# Display result
echo "Repository push completed. Check https://github.com/CasapeStock/MonteCarlo"
