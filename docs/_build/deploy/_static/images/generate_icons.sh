#!/bin/bash

# YICA/YiRage Icon Generation Script
# This script generates real PNG and ICO files from SVG sources

echo "🎨 Generating YICA/YiRage icons..."

# Check if required tools are available
if ! command -v inkscape &> /dev/null; then
    echo "⚠️  Inkscape not found. Please install inkscape to generate PNG files."
    echo "   macOS: brew install inkscape"
    echo "   Ubuntu: sudo apt-get install inkscape"
    echo "   Windows: Download from https://inkscape.org/"
fi

if ! command -v convert &> /dev/null; then
    echo "⚠️  ImageMagick not found. Please install imagemagick to generate ICO files."
    echo "   macOS: brew install imagemagick"
    echo "   Ubuntu: sudo apt-get install imagemagick"
    echo "   Windows: Download from https://imagemagick.org/"
fi

# Generate PNG logo from SVG
if command -v inkscape &> /dev/null; then
    echo "📸 Generating PNG logo..."
    inkscape --export-filename=yica-logo.png --export-width=200 --export-height=80 yica-logo.svg
    echo "✅ Generated yica-logo.png"
else
    echo "⏭️  Skipping PNG generation (inkscape not available)"
fi

# Generate favicon PNG from SVG
if command -v inkscape &> /dev/null; then
    echo "📸 Generating favicon PNG..."
    inkscape --export-filename=favicon-32.png --export-width=32 --export-height=32 favicon.svg
    echo "✅ Generated favicon-32.png"
else
    echo "⏭️  Skipping favicon PNG generation (inkscape not available)"
fi

# Generate ICO from PNG
if command -v convert &> /dev/null && [ -f "favicon-32.png" ]; then
    echo "🔄 Converting PNG to ICO..."
    convert favicon-32.png -define icon:auto-resize=16,32 favicon.ico
    echo "✅ Generated favicon.ico"
    rm favicon-32.png  # Clean up temporary file
else
    echo "⏭️  Skipping ICO generation (imagemagick not available or PNG not found)"
fi

echo ""
echo "🎉 Icon generation complete!"
echo ""
echo "📁 Generated files:"
if [ -f "yica-logo.png" ]; then
    echo "   ✅ yica-logo.png (200x80)"
fi
if [ -f "favicon.ico" ]; then
    echo "   ✅ favicon.ico (16x16, 32x32)"
fi
echo ""
echo "🌐 Alternative online tools if local generation fails:"
echo "   • PNG: https://convertio.co/svg-png/"
echo "   • ICO: https://favicon.io/favicon-converter/"
echo ""
echo "📝 Manual steps:"
echo "   1. Upload yica-logo.svg to online converter"
echo "   2. Download as PNG (200x80)"
echo "   3. Upload favicon.svg to favicon converter"
echo "   4. Download as ICO (32x32)"
echo "   5. Replace placeholder files with real binary files"
