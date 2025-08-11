YICA-YiRage Documentation
=========================

Welcome to YICA-YiRage: AI Computing Optimization Framework for In-Memory Computing Architecture.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/README
   getting-started/quick-reference

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   USAGE
   tutorials/README

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/README
   api/python-api-corrected
   api/cpp-api-verified
   api/analyzer

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/README

Index and Tables
================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

What is YICA-YiRage?
====================

**YICA-YiRage** is an AI computing optimization framework specifically designed for in-memory computing architectures. It extends the YiRage superoptimization engine with YICA (Yet another In-memory Computing Architecture) support, providing automated GPU kernel generation and optimization for deep learning workloads on specialized hardware.

Key Features
============

* **🚀 Automated Kernel Generation**: Automatically generates optimized GPU kernels without manual CUDA/Triton programming
* **🧠 In-Memory Computing Support**: Specialized optimizations for in-memory computing architectures 
* **⚡ Superoptimization**: Multi-level optimization techniques for maximum performance
* **🔄 PyTorch Integration**: Seamless integration with existing PyTorch workflows
* **🎯 Production Ready**: Comprehensive testing and validation framework
* **📊 Performance Monitoring**: Built-in profiling and performance analysis tools

Quick Start
===========

.. code-block:: python

   import yirage as yr

   # Create a kernel graph
   graph = yr.new_kernel_graph()

   # Define input tensors
   X = graph.new_input(dims=(1024, 512), dtype=yr.float16)
   W = graph.new_input(dims=(512, 256), dtype=yr.float16)

   # Add operations
   Y = graph.rms_norm(X, normalized_shape=(512,))
   Z = graph.matmul(Y, W)

   # Mark outputs
   graph.mark_output(Z)

   # Generate optimized kernel
   kernel = graph.superoptimize()

   # Use in PyTorch
   import torch
   x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
   w = torch.randn(512, 256, dtype=torch.float16, device='cuda')
   output = kernel(inputs=[x, w])

Installation
============

.. code-block:: bash

   # From PyPI (Recommended)
   pip install yica-yirage

   # From Source
   git clone --recursive https://github.com/yica-ai/yica-yirage.git
   cd yica-yirage
   pip install -e . -v

Getting Help
============

If you encounter issues:

1. Check the :doc:`User Guide <USAGE>`
2. Browse :doc:`API Reference <api/README>`
3. View :doc:`Quick Reference <getting-started/quick-reference>`
4. Submit an Issue on `GitHub <https://github.com/yica-ai/yica-yirage/issues>`_
