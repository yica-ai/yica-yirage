# Contributing to YICA-Yirage

We welcome contributions to YICA-Yirage! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yicai-ai/yica-yirage.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Create a feature branch: `git checkout -b feature/amazing-feature`

## Code Style

- Follow PEP 8 for Python code
- Use Black for code formatting: `black yirage/python/`
- Use isort for import sorting: `isort yirage/python/`
- Run type checking: `mypy yirage/python/`

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
