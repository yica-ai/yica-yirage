# Documentation Usage

## Quick Start

```bash
# Build documentation
cd docs
python3 -m venv docs_env
source docs_env/bin/activate  # Linux/macOS
pip install -r requirements-simple.txt
make build

# View locally
make serve
# Open http://localhost:8000
```

## Deploy

```bash
# Deploy to Read the Docs
./deploy.sh --deploy

# Or manually
rtd deploy yica-yirage
```

## Access

- **Local**: http://localhost:8000
- **Read the Docs**: https://yica-yirage.readthedocs.io/
- **GitHub Pages**: https://your-username.github.io/YZ-optimzier-bin/
