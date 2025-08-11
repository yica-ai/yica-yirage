#!/bin/bash

# YICA/YiRage Documentation Deployment Script

set -e

echo "🚀 Starting YICA/YiRage documentation deployment..."

# Check if we're in the docs directory
if [ ! -f "conf.py" ]; then
    echo "❌ Error: Please run this script from the docs directory"
    exit 1
fi

# Build documentation
echo "📚 Building documentation..."
make clean
make build

echo "✅ Documentation built successfully!"

# Check if we should deploy
if [ "$1" = "--deploy" ]; then
    echo "🌐 Deploying to Read the Docs..."
    
    # Check if rtd-cli is installed
    if ! command -v rtd &> /dev/null; then
        echo "📦 Installing Read the Docs CLI..."
        pip install rtd-cli
    fi
    
    # Deploy
    rtd deploy yica-yirage
    
    echo "✅ Documentation deployed to https://yica-yirage.readthedocs.io/"
else
    echo "📖 Documentation built in _build/html/"
    echo "🌐 To deploy, run: $0 --deploy"
    echo "🔗 Or visit: http://localhost:8000 (run 'make serve' to start local server)"
fi
