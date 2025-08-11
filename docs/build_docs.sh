#!/bin/bash

# YICA/YiRage Documentation Build Script
# This script builds the documentation for YICA/YiRage project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Python version: $PYTHON_VERSION"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.8+ is required"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing documentation dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Function to setup documentation environment
setup_docs() {
    print_status "Setting up documentation environment..."
    
    # Create necessary directories
    mkdir -p _static/css _static/js _static/images _templates
    
    # Create placeholder files if they don't exist
    [ -f "_static/css/custom.css" ] || touch _static/css/custom.css
    [ -f "_static/js/custom.js" ] || touch _static/js/custom.js
    
    print_success "Documentation environment setup complete"
}

# Function to build documentation
build_docs() {
    print_status "Building documentation..."
    
    # Clean previous build
    make clean
    
    # Build HTML documentation
    make build
    
    print_success "Documentation built successfully"
}

# Function to test documentation
test_docs() {
    print_status "Testing documentation..."
    
    # Check for common issues
    if [ -d "_build/html" ]; then
        print_success "HTML build directory exists"
        
        # Check for index.html
        if [ -f "_build/html/index.html" ]; then
            print_success "Main index.html exists"
        else
            print_warning "index.html not found"
        fi
        
        # Check build size
        BUILD_SIZE=$(du -sh _build/html | cut -f1)
        print_status "Build size: $BUILD_SIZE"
        
    else
        print_error "HTML build directory not found"
        exit 1
    fi
}

# Function to serve documentation locally
serve_docs() {
    print_status "Starting local documentation server..."
    print_status "Documentation will be available at: http://localhost:8000"
    print_status "Press Ctrl+C to stop the server"
    
    cd _build/html
    python3 -m http.server 8000
}

# Function to build PDF documentation
build_pdf() {
    print_status "Building PDF documentation..."
    
    # Check if LaTeX is available
    if command_exists pdflatex; then
        make build-pdf
        print_success "PDF documentation built successfully"
    else
        print_warning "LaTeX not found, skipping PDF build"
        print_status "Install LaTeX to build PDF documentation"
    fi
}

# Function to show help
show_help() {
    echo "YICA/YiRage Documentation Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -i, --install   Install dependencies only"
    echo "  -b, --build     Build documentation only"
    echo "  -t, --test      Test documentation only"
    echo "  -s, --serve     Serve documentation locally"
    echo "  -p, --pdf       Build PDF documentation"
    echo "  -a, --all       Full build process (default)"
    echo "  -c, --clean     Clean build directory"
    echo ""
    echo "Examples:"
    echo "  $0              # Full build process"
    echo "  $0 --serve      # Build and serve locally"
    echo "  $0 --pdf        # Build PDF documentation"
    echo "  $0 --clean      # Clean build directory"
}

# Function to clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    make clean
    print_success "Build directory cleaned"
}

# Main script logic
main() {
    print_status "Starting YICA/YiRage documentation build process..."
    
    # Parse command line arguments
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--install)
            check_python_version
            install_dependencies
            setup_docs
            exit 0
            ;;
        -b|--build)
            build_docs
            exit 0
            ;;
        -t|--test)
            test_docs
            exit 0
            ;;
        -s|--serve)
            build_docs
            serve_docs
            exit 0
            ;;
        -p|--pdf)
            build_pdf
            exit 0
            ;;
        -c|--clean)
            clean_build
            exit 0
            ;;
        -a|--all|"")
            # Full build process (default)
            check_python_version
            install_dependencies
            setup_docs
            build_docs
            test_docs
            print_success "Documentation build process completed successfully!"
            print_status "You can now:"
            print_status "  - View the documentation at: _build/html/index.html"
            print_status "  - Serve it locally with: $0 --serve"
            print_status "  - Build PDF with: $0 --pdf"
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
