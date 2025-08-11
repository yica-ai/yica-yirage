#!/bin/bash

# Manual deployment script for GitHub Pages
set -e

echo "Building documentation..."
make build

echo "Creating gh-pages branch..."
cd _build/html

# Initialize git repository if not exists
if [ ! -d .git ]; then
    git init
    git remote add origin https://github.com/yica-ai/yica-mirage.git
fi

# Add all files
git add -A

# Commit changes
git commit -m "Deploy documentation to GitHub Pages" || echo "No changes to commit"

# Push to gh-pages branch
echo "Pushing to gh-pages branch..."
git push origin HEAD:gh-pages --force

echo "Deployment completed!"
echo "Documentation will be available at: https://yica-ai.github.io/yica-mirage/"
