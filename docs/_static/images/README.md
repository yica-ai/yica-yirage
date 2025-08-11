# YICA/YiRage Icons and Graphics

This directory contains the visual assets for the YICA/YiRage documentation.

## 📁 Files

### SVG Source Files (Vector Graphics)
- **`yica-logo.svg`** - Main YICA logo with YiRage branding (200x80)
- **`favicon.svg`** - Favicon source file (32x32)

### Generated Files (Placeholders)
- **`yica-logo.png`** - Main logo in PNG format (placeholder)
- **`favicon.ico`** - Website favicon (placeholder)

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
The files are automatically used by Sphinx configuration:

```python
# In docs/conf.py
html_logo = '_static/images/yica-logo.png'
html_favicon = '_static/images/favicon.ico'
```

### In Markdown Files
```markdown
![YICA Logo](/_static/images/yica-logo.png)
```

### In HTML
```html
<img src="_static/images/yica-logo.png" alt="YICA Logo" width="200" height="80">
<link rel="icon" type="image/x-icon" href="_static/images/favicon.ico">
```

## 🔄 Generating Real Files

### Option 1: Local Generation (Recommended)
```bash
# Install required tools
# macOS:
brew install inkscape imagemagick

# Ubuntu/Debian:
sudo apt-get install inkscape imagemagick

# Run generation script
./generate_icons.sh
```

### Option 2: Online Conversion
1. **For PNG Logo**:
   - Upload `yica-logo.svg` to https://convertio.co/svg-png/
   - Set width to 200px, height to 80px
   - Download and replace `yica-logo.png`

2. **For Favicon**:
   - Upload `favicon.svg` to https://favicon.io/favicon-converter/
   - Generate ICO with 16x16 and 32x32 sizes
   - Download and replace `favicon.ico`

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
| `yica-logo.svg` | ✅ Ready | Vector | SVG | Source file |
| `favicon.svg` | ✅ Ready | Vector | SVG | Source file |
| `yica-logo.png` | ⚠️ Placeholder | 200x80 | PNG | Documentation |
| `favicon.ico` | ⚠️ Placeholder | 32x32 | ICO | Browser |

## 🔧 Next Steps

To complete the icon setup:

1. **Generate real binary files** using the provided script or online tools
2. **Test favicon display** in browser
3. **Verify logo display** in documentation
4. **Commit binary files** to repository
5. **Deploy to production** platforms

---

**Note**: The current PNG and ICO files are text placeholders. Use the generation script or online tools to create actual binary image files for production use.
