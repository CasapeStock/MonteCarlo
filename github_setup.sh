#!/bin/bash

# Prompt for GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Prompt for repository name
read -p "Enter repository name (default: StockControl): " REPO_NAME
REPO_NAME=${REPO_NAME:-StockControl}

# Prompt for repository description
read -p "Enter repository description: " REPO_DESC

echo "Manual GitHub Repository Setup Instructions:"
echo "1. Go to https://github.com/new and create a new repository"
echo "   - Name: $REPO_NAME"
echo "   - Description: $REPO_DESC"
echo "   - Visibility: Public"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. After creating the repository, run these commands:"
echo "   git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Press Enter when you have created the repository on GitHub..."
read

# Verify git remote setup
git remote -v
