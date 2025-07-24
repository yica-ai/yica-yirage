#!/bin/bash

# YICA-Mirage GitHub Repository Setup Script
# Creates and configures the GitHub repository for public release

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# Repository configuration
REPO_OWNER="yica-ai"
REPO_NAME="yica-mirage"
REPO_DESCRIPTION="YICA-Mirage: High-Performance AI Computing Optimization Framework for In-Memory Computing Architecture"
REPO_URL="https://github.com/$REPO_OWNER/$REPO_NAME"

log_info "=========================================="
log_info "YICA-Mirage GitHub Repository Setup"
log_info "Repository: $REPO_URL"
log_info "=========================================="

# Check if GitHub CLI is installed
if ! command -v gh >/dev/null 2>&1; then
    log_error "GitHub CLI (gh) is not installed. Please install it first:"
    log_info "  macOS: brew install gh"
    log_info "  Linux: sudo apt install gh"
    log_info "  Windows: winget install GitHub.cli"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status >/dev/null 2>&1; then
    log_warn "Not authenticated with GitHub. Running authentication..."
    gh auth login
fi

# Create repository if it doesn't exist
log_info "Creating GitHub repository..."
if gh repo view "$REPO_OWNER/$REPO_NAME" >/dev/null 2>&1; then
    log_warn "Repository already exists. Skipping creation."
else
    gh repo create "$REPO_OWNER/$REPO_NAME" \
        --description "$REPO_DESCRIPTION" \
        --homepage "https://yica.ai" \
        --public \
        --clone=false
    log_info "Repository created successfully!"
fi

# Configure repository settings
log_info "Configuring repository settings..."

# Enable features
gh repo edit "$REPO_OWNER/$REPO_NAME" \
    --enable-issues \
    --enable-projects \
    --enable-wiki \
    --enable-discussions

# Set repository topics
gh repo edit "$REPO_OWNER/$REPO_NAME" \
    --add-topic "ai" \
    --add-topic "optimization" \
    --add-topic "compiler" \
    --add-topic "triton" \
    --add-topic "yica" \
    --add-topic "mirage" \
    --add-topic "deep-learning" \
    --add-topic "in-memory-computing" \
    --add-topic "machine-learning" \
    --add-topic "gpu-computing"

log_info "Repository settings configured!"

# Setup branch protection (if repository has content)
setup_branch_protection() {
    log_info "Setting up branch protection rules..."
    
    # Create branch protection rule for main branch
    gh api repos/$REPO_OWNER/$REPO_NAME/branches/main/protection \
        --method PUT \
        --field required_status_checks='{"strict":true,"contexts":["build-and-test"]}' \
        --field enforce_admins=true \
        --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
        --field restrictions=null \
        2>/dev/null || log_warn "Branch protection setup failed (repository may be empty)"
}

# Setup repository secrets (you'll need to add these manually or via API)
setup_secrets() {
    log_info "Repository secrets that need to be configured:"
    log_warn "Please add these secrets manually in GitHub repository settings:"
    echo "  - PYPI_API_TOKEN: PyPI API token for package publishing"
    echo "  - DOCKER_USERNAME: Docker Hub username"
    echo "  - DOCKER_PASSWORD: Docker Hub password or token"
    echo ""
    log_info "Navigate to: $REPO_URL/settings/secrets/actions"
}

# Create initial repository structure
create_initial_structure() {
    log_info "Preparing initial repository structure..."
    
    # Create essential files that might be missing
    cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 YICA Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

    cat > CONTRIBUTING.md << 'EOF'
# Contributing to YICA-Mirage

We welcome contributions to YICA-Mirage! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/yica-mirage.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature/amazing-feature`

## Code Style

- Follow PEP 8 for Python code
- Use Black for code formatting: `black mirage/python/`
- Use isort for import sorting: `isort mirage/python/`
- Run type checking: `mypy mirage/python/`

## Testing

- Run tests: `python -m pytest tests/ -v`
- Ensure all tests pass before submitting PR
- Add tests for new functionality

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure CI passes
4. Request review from maintainers

## Code of Conduct

Be respectful and inclusive in all interactions.
EOF

    cat > CHANGELOG.md << 'EOF'
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-24

### Added
- Initial release of YICA-Mirage
- Mirage-based universal code optimization
- YICA in-memory computing architecture support
- Automatic Triton code generation
- Multi-backend support (CPU/GPU/YICA)
- Python bindings and command-line tools
- Cross-platform installation support
- Comprehensive documentation and examples

### Features
- High-performance AI computing optimization framework
- Specialized optimizations for in-memory computing architectures
- Intelligent performance tuning with advanced search algorithms
- Full CUDA compatibility and backward support
- Easy-to-use Python API with C++ performance
EOF

    log_info "Essential files created!"
}

# Git operations
setup_git_repo() {
    log_info "Setting up Git repository..."
    
    # Initialize git if not already initialized
    if [ ! -d ".git" ]; then
        git init
        log_info "Git repository initialized"
    fi
    
    # Add remote if not exists
    if ! git remote get-url origin >/dev/null 2>&1; then
        git remote add origin "$REPO_URL.git"
        log_info "Remote origin added"
    fi
    
    # Create .gitignore if not exists
    if [ ! -f ".gitignore" ]; then
        log_warn ".gitignore not found. Please ensure it exists."
    fi
    
    # Set up git configuration
    git config --local user.name "YICA Team"
    git config --local user.email "contact@yica.ai"
    
    log_info "Git repository configured!"
}

# Push to GitHub
push_to_github() {
    log_info "Preparing to push to GitHub..."
    
    # Stage all files
    git add .
    
    # Check if there are changes to commit
    if git diff --staged --quiet; then
        log_warn "No changes to commit"
        return
    fi
    
    # Commit changes
    git commit -m "Initial release: YICA-Mirage v1.0.0

- High-performance AI computing optimization framework
- YICA in-memory computing architecture support
- Automatic Triton code generation
- Multi-backend support (CPU/GPU/YICA)
- Cross-platform installation support
- Complete documentation and examples"
    
    # Create and push main branch
    git branch -M main
    git push -u origin main
    
    # Create and push initial tag
    git tag -a v1.0.0 -m "YICA-Mirage v1.0.0 - Initial Release"
    git push origin v1.0.0
    
    log_info "Code pushed to GitHub successfully!"
}

# Create release
create_github_release() {
    log_info "Creating GitHub release..."
    
    gh release create v1.0.0 \
        --repo "$REPO_OWNER/$REPO_NAME" \
        --title "YICA-Mirage v1.0.0" \
        --notes "🎉 **Initial Release of YICA-Mirage**

## What's New
- High-performance AI computing optimization framework
- YICA in-memory computing architecture support  
- Automatic Triton code generation
- Multi-backend support (CPU/GPU/YICA)
- Intelligent performance tuning
- Cross-platform installation support

## Installation

### pip (Recommended)
\`\`\`bash
pip install yica-mirage
\`\`\`

### Homebrew (macOS)
\`\`\`bash
brew tap yica-ai/tap
brew install yica-mirage
\`\`\`

### Docker
\`\`\`bash
docker run -it yicaai/yica-mirage:cpu-latest
\`\`\`

## Documentation
- [API Reference](https://yica-mirage.readthedocs.io/)
- [Architecture Guide](docs/architecture/)
- [Examples](examples/)

Full changelog and documentation available at https://github.com/yica-ai/yica-mirage" \
        --latest
    
    log_info "GitHub release created!"
}

# Main execution
main() {
    log_info "Starting repository setup process..."
    
    # Create essential files
    create_initial_structure
    
    # Setup git repository
    setup_git_repo
    
    # Push to GitHub
    push_to_github
    
    # Setup branch protection
    setup_branch_protection
    
    # Create release
    create_github_release
    
    # Show next steps
    log_info "=========================================="
    log_info "Repository setup completed successfully!"
    log_info "=========================================="
    log_info "Next steps:"
    log_info "1. Visit: $REPO_URL"
    log_info "2. Configure repository secrets:"
    setup_secrets
    log_info "3. Enable GitHub Actions workflows"
    log_info "4. Set up package repositories (PyPI, etc.)"
    log_info "5. Configure documentation hosting"
    log_info ""
    log_info "Repository is now ready for public release! 🚀"
}

# Run main function
main "$@" 