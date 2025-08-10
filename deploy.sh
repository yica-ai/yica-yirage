#!/bin/bash

# YICA/YiRage 文档部署脚本
# 支持多种部署平台：Netlify, Vercel, GitHub Pages

set -e

echo "🚀 YICA/YiRage 文档部署脚本"
echo "=============================="

# 检查参数
PLATFORM=${1:-"static"}
BUILD_DIR="dist"

case $PLATFORM in
  "netlify")
    echo "📡 部署到Netlify..."
    npm run build
    echo "✅ 构建完成，请将 .output/public 目录上传到Netlify"
    ;;
    
  "vercel")
    echo "▲ 部署到Vercel..."
    npm run build
    echo "✅ 构建完成，请使用 vercel deploy 命令部署"
    ;;
    
  "github-pages")
    echo "🐙 部署到GitHub Pages..."
    npm run build
    
    # 创建GitHub Pages分支
    if git show-ref --verify --quiet refs/heads/gh-pages; then
      git checkout gh-pages
      git pull origin gh-pages
    else
      git checkout --orphan gh-pages
    fi
    
    # 清理旧文件
    git rm -rf . --quiet || true
    
    # 复制构建文件
    cp -r .output/public/* .
    
    # 添加.nojekyll文件
    touch .nojekyll
    
    # 提交更改
    git add .
    git commit -m "Deploy docs: $(date)"
    git push origin gh-pages
    
    echo "✅ 已部署到GitHub Pages"
    ;;
    
  "static")
    echo "📦 生成静态文件..."
    npm run build
    
    if [ -d ".output/public" ]; then
      rm -rf $BUILD_DIR
      cp -r .output/public $BUILD_DIR
      echo "✅ 静态文件已生成到 $BUILD_DIR 目录"
      echo "📁 文件列表:"
      ls -la $BUILD_DIR
    else
      echo "❌ 构建失败：未找到输出目录"
      exit 1
    fi
    ;;
    
  "docker")
    echo "🐳 构建Docker镜像..."
    
    # 创建Dockerfile
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

    # 创建nginx配置
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
        
        # 启用gzip压缩
        gzip on;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    }
}
EOF

    # 构建Docker镜像
    docker build -t yica-yirage-docs:latest .
    
    echo "✅ Docker镜像构建完成"
    echo "🚀 运行命令: docker run -p 8080:80 yica-yirage-docs:latest"
    ;;
    
  *)
    echo "❌ 未知的部署平台: $PLATFORM"
    echo ""
    echo "支持的部署平台:"
    echo "  static        - 生成静态文件"
    echo "  netlify       - 部署到Netlify"
    echo "  vercel        - 部署到Vercel"
    echo "  github-pages  - 部署到GitHub Pages"
    echo "  docker        - 构建Docker镜像"
    echo ""
    echo "使用方法: $0 [platform]"
    exit 1
    ;;
esac

echo ""
echo "🎉 部署完成！"
