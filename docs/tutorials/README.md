# Tutorials and Learning Resources

This directory contains comprehensive tutorials, examples, and learning resources for YICA/YiRage.

## 🎯 Executive Presentations & Demos

### For Leadership Review
- **[Executive Overview](yica-yirage-executive-overview.md)** - Complete executive briefing with business value, ROI analysis, and strategic positioning
- **[Live Demo Script](yica_working_demo.py)** - Runnable demonstration showing actual YICA-YiRage capabilities
- **[Technical Demo](yica_live_demo.py)** - Comprehensive technical demonstration with all features

### Quick Demo Command
```bash
# Run the working demo to see YICA-YiRage in action
python docs/tutorials/yica_working_demo.py
```

## 📚 Tutorial Overview

### Getting Started Tutorials
- **Installation Tutorial** - Step-by-step installation guide
- **First Optimization** - Your first model optimization
- **Understanding Results** - Interpreting optimization results

### Intermediate Tutorials
- **Multi-Backend Optimization** - Comparing different backends
- **Custom Operators** - Creating and optimizing custom operators
- **Performance Tuning** - Advanced performance optimization techniques

### Advanced Tutorials
- **[Performance Benchmarks](performance-benchmarks.md)** - Comprehensive performance analysis and benchmarking
- **[Real-World Examples](real-world-examples.md)** - Source code based practical examples and implementations
- **Backend Development** - Creating custom optimization backends
- **Integration Patterns** - Integrating YiRage with existing workflows
- **Production Deployment** - Deploying optimized models in production

## 🎯 Learning Paths

### Beginner Path
```
Installation → First Optimization → Understanding Results → Basic Examples
```

### Developer Path
```
Architecture Overview → API Learning → Advanced Features → Custom Development
```

### DevOps Path
```
Installation → Container Deployment → Monitoring → Production Best Practices
```

### Researcher Path
```
Theoretical Foundation → Algorithm Principles → Performance Analysis → Research
```

## 🚀 Quick Start Tutorial

### Step 1: Installation
```bash
# Install via pip
pip install yirage

# Or build from source
git clone https://github.com/yica-ai/yica-yirage.git
cd yica-yirage && pip install -e .
```

### Step 2: Basic Usage
```python
import yirage
import torch

# Create a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Optimize with YiRage
optimizer = yirage.Optimizer(backend="yica")
optimized_model = optimizer.optimize(model)

print(f"Speedup: {optimized_model.speedup:.2f}x")
```

### Step 3: Verify Results
```python
test_input = torch.randn(1, 784)
original_output = model(test_input)
optimized_output = optimized_model(test_input)

# Verify correctness
assert torch.allclose(original_output, optimized_output, atol=1e-5)
```

## 📖 Tutorial Categories

### 1. Basic Tutorials
- **Installation and Setup**: System requirements and installation
- **First Steps**: Hello world and basic API usage
- **Configuration**: Basic configuration options
- **Troubleshooting**: Common issues and solutions

### 2. Core Concepts
- **Architecture Understanding**: YICA architecture overview
- **Backend Comparison**: Understanding different backends
- **Optimization Strategies**: Different optimization approaches

### 3. Practical Examples
- **Computer Vision**: Image classification and object detection
- **Natural Language Processing**: Text processing and generation
- **Large Language Models**: LLM optimization techniques

### 4. Advanced Topics
- **Custom Backend Development**: Creating new backends
- **Performance Analysis**: Detailed profiling and analysis
- **Production Integration**: Real-world deployment patterns

## 🛠️ Interactive Learning

### Jupyter Notebooks
```bash
pip install jupyter
jupyter notebook tutorials/
```

### Docker Tutorial Environment
```bash
docker run -it --rm -p 8888:8888 yirage/tutorial-environment
```

## 📊 Example Projects

### Image Classification Optimization
```python
import torchvision
import yirage

model = torchvision.models.efficientnet_b0(pretrained=True)
optimizer = yirage.Optimizer(backend="yica")
optimized_model = optimizer.optimize(model)

print(f"Performance improvement: {optimized_model.speedup:.2f}x")
```

### Natural Language Processing
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
optimizer = yirage.Optimizer(backend="yica")
optimized_model = optimizer.optimize(model)
```

## 🎓 Learning Resources

### Documentation Links
- [Architecture Overview](../architecture/README.md)
- [API Reference](../api/README.md)
- [Development Guide](../development/README.md)
- [Deployment Guide](../deployment/README.md)

### Community Resources
- **GitHub Discussions**: Technical Q&A
- **Discord Server**: Real-time community chat
- **YouTube Channel**: Video tutorials
- **Blog**: Technical articles and case studies

## 📋 Progress Tracking

### Beginner Level
- [ ] Complete installation
- [ ] Run first optimization
- [ ] Understand basic concepts
- [ ] Complete simple project

### Intermediate Level
- [ ] Multi-backend comparison
- [ ] Custom operator optimization
- [ ] Performance analysis
- [ ] Workflow integration

### Advanced Level
- [ ] Custom backend development
- [ ] Production deployment
- [ ] Performance research
- [ ] Project contribution

## 📞 Support

### Getting Help
- **GitHub Issues**: Report tutorial problems
- **Community Support**: Discord and GitHub Discussions
- **Documentation**: Comprehensive guides
- **Examples**: Repository example code

---

*These tutorials help users master YICA/YiRage at all levels. Check our [GitHub repository](https://github.com/yica-ai/yica-yirage) for the latest updates.*