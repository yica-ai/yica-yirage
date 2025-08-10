#!/bin/bash

# YICA/YiRage Documentation Deployment Script
# Supports multiple deployment platforms: Netlify, Vercel, GitHub Pages

set -e

echo "🚀 YICA/YiRage Documentation Deployment Script"
echo "=============================================="

# Check parameters
PLATFORM=${1:-"static"}
BUILD_DIR="dist"

case $PLATFORM in
  "netlify")
    echo "📡 Deploying to Netlify..."
    npm run build
    echo "✅ Build completed, please upload .output/public directory to Netlify"
    ;;
    
  "vercel")
    echo "▲ Deploying to Vercel..."
    npm run build
    echo "✅ Build completed, please use vercel deploy command to deploy"
    ;;
    
  "github-pages")
    echo "🐙 Deploying to GitHub Pages..."
    npm run build
    
    # Create GitHub Pages branch
    if git show-ref --verify --quiet refs/heads/gh-pages; then
      git checkout gh-pages
      git pull origin gh-pages
    else
      git checkout --orphan gh-pages
    fi
    
    # Clean old files
    git rm -rf . --quiet || true
    
    # Copy build files
    cp -r .output/public/* .
    
    # Add .nojekyll file
    touch .nojekyll
    
    # Commit changes
    git add .
    git commit -m "Deploy docs: $(date)"
    git push origin gh-pages
    
    echo "✅ Deployed to GitHub Pages"
    ;;
    
  "static")
    echo "📦 Generating static files..."
    npm run build
    
    if [ -d ".output/public" ]; then
      rm -rf $BUILD_DIR
      cp -r .output/public $BUILD_DIR
      echo "✅ Static files generated to $BUILD_DIR directory"
      echo "📁 File list:"
      ls -la $BUILD_DIR
    else
      echo "❌ Build failed: Output directory not found"
      exit 1
    fi
    ;;
    
  "docker")
    echo "🐳 Building Docker image..."
    
    # Create Dockerfile
    cat > Dockerfile << EOF
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/.output/public /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

    # Create nginx configuration
    cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            root /usr/share/nginx/html;
            index index.html;
            try_files \$uri \$uri/ /index.html;
        }
        
        # Enable gzip compression
        gzip on;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    }
}
EOF

    # Build Docker image
    docker build -t yica-yirage-docs:latest .
    
    echo "✅ Docker image build completed"
    echo "🚀 Run command: docker run -p 8080:80 yica-yirage-docs:latest"
    ;;
    
  *)
    echo "❌ Unknown deployment platform: $PLATFORM"
    echo ""
    echo "Supported deployment platforms:"
    echo "  static        - Generate static files"
    echo "  netlify       - Deploy to Netlify"
    echo "  vercel        - Deploy to Vercel"
    echo "  github-pages  - Deploy to GitHub Pages"
    echo "  docker        - Build Docker image"
    echo ""
    echo "Usage: $0 [platform]"
    exit 1
    ;;
esac

echo ""
echo "🎉 Deployment completed!"
