# YICA/YiRage Icons and Graphics

This directory contains the visual assets for the YICA/YiRage documentation.

## 📁 Files

### SVG Source Files (Vector Graphics)
- **`yica-logo.svg`** - Main YICA logo with YiRage branding (200x80)
- **`favicon.svg`** - Favicon source file (32x32)

### Production Files
- **SVG files are used directly** - No conversion needed for modern browsers

### Utilities
- **`generate_icons.sh`** - Script to generate real PNG/ICO files from SVG sources
- **`README.md`** - This documentation file

## 🎨 Logo Design

### Main Logo (`yica-logo.svg`)
- **YICA** text in gradient blue
- **CIM Array** grid representation showing memory cells
- **YiRage** optimizer branding
- **Performance indicators** (arrows, "10x Faster")
- **Modern tech aesthetic** with clean lines
- **Size**: 200x80 pixels
- **Colors**: Blue gradient (#2980B9 to #5DADE2)

### Favicon (`favicon.svg`)
- **Stylized "Y"** for YICA in white
- **Blue gradient background** circle
- **CIM array dots** representation
- **Green performance arrow**
- **Size**: 32x32 pixels
- **Optimized** for small display sizes

## 🛠️ Usage

### In Sphinx Documentation
The SVG files are automatically used by Sphinx configuration:

```python
# In docs/conf.py
html_logo = '_static/images/yica-logo.svg'
html_favicon = '_static/images/favicon.svg'
```

### In Markdown Files
```markdown
![YICA Logo](/_static/images/yica-logo.svg)
```

### In HTML
```html
<img src="_static/images/yica-logo.svg" alt="YICA Logo" width="200" height="80">
<link rel="icon" type="image/svg+xml" href="_static/images/favicon.svg">
```

## ✅ SVG Direct Usage

### Modern Browser Support
SVG files are used directly without conversion:
- **Scalable**: Perfect quality at any size
- **Small file size**: Vector graphics are efficient
- **Modern standard**: Supported by all current browsers
- **No conversion needed**: Ready for production use

### Optional PNG/ICO Generation
If you need raster formats for legacy support:

```bash
# Install required tools (optional)
# macOS:
brew install inkscape imagemagick

# Ubuntu/Debian:
sudo apt-get install inkscape imagemagick

# Run generation script (optional)
./generate_icons.sh
```

### Option 3: Manual Creation
Use any graphics editor that supports SVG:
- **Adobe Illustrator**
- **Inkscape** (free)
- **Figma** (online)
- **Canva** (online)

## 🎯 Brand Guidelines

### Colors
- **Primary Blue**: #2980B9
- **Secondary Blue**: #3498DB  
- **Light Blue**: #5DADE2
- **Success Green**: #27AE60
- **Dark Gray**: #34495E
- **Light Gray**: #7F8C8D

### Typography
- **Main Font**: Arial, sans-serif
- **Logo Text**: Bold, gradient blue
- **Subtitle**: Regular, gray

### Usage Rules
1. **Maintain aspect ratio** when resizing
2. **Preserve minimum size** (logo: 100px width minimum)
3. **Use on light backgrounds** for best contrast
4. **Don't modify colors** without approval
5. **Include clear space** around logo (minimum 20px)

## 🌐 Deployment

These assets are automatically deployed with the documentation to:
- **GitHub Pages**: `https://yica-ai.github.io/yica-yirage/`
- **Read the Docs**: `https://yica-yirage.readthedocs.io/`
- **Netlify**: Custom domain deployment

## 📝 File Status

| File | Status | Size | Format | Usage |
|------|--------|------|--------|-------|
| `yica-logo.svg` | ✅ Production Ready | Vector | SVG | Documentation logo |
| `favicon.svg` | ✅ Production Ready | Vector | SVG | Browser favicon |

## 🔧 Ready for Production

The SVG icons are production-ready:

1. ✅ **SVG files created** with professional design
2. ✅ **Sphinx configuration updated** to use SVG directly
3. ✅ **Modern browser support** for SVG favicons and logos
4. ✅ **Scalable graphics** that look perfect at any size
5. ✅ **Ready for deployment** to all platforms

---

**Status**: ✅ **Production Ready** - SVG icons are immediately usable in all modern browsers and documentation platforms.
