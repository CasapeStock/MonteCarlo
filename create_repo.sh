#!/bin/bash

# Prompt for repository name
read -p "Enter new repository name (e.g., StockControlApp): " REPO_NAME

# Create a new repository on GitHub
echo "Creating new repository $REPO_NAME on GitHub..."
echo "Please enter your GitHub password when prompted"

# Use curl to create a new repository via GitHub API
curl -u "CasapeStock" https://api.github.com/user/repos -d "{\"name\": \"$REPO_NAME\", \"description\": \"Stock Evaluation Mobile and Web Application\", \"private\": false}"

# Set up local repository
git init
git add .
git commit -m "Initial commit: Stock Evaluation App setup"
git branch -M main

# Add remote repository
git remote add origin "https://github.com/CasapeStock/$REPO_NAME.git"

# Push to GitHub
git push -u origin main

echo "Repository $REPO_NAME created and code pushed successfully!"
