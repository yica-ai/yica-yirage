YICA/YiRage Documentation
=========================

Welcome to the YICA (YICA Intelligence Computing Architecture) and YiRage (AI Kernel Super Optimizer) documentation.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/README
   getting-started/design-philosophy
   getting-started/quick-reference

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/README
   architecture/yirage-architecture
   architecture/yica-architecture-detailed
   architecture/modular-architecture
   architecture/implementation-summary
   architecture/yirage-integration-plan
   architecture/yirage-extension
   architecture/yirage-updates

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
   design/build_system_redesign
   design/compatibility_layer_enhancement
   design/configuration_management_system
   design/deployment_packaging_strategy
   design/error_handling_logging_system
   design/testing_framework_design

.. toctree::
   :maxdepth: 2
   :caption: Deployment

   deployment/README
   deployment/docker-deployment
   deployment/deployment-report

.. toctree::
   :maxdepth: 2
   :caption: Project Management

   project-management/README
   project-management/backend-integration
   project-management/implementation-analysis
   project-management/execution-plan

Index and Tables
================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Project Overview
================

YICA (YICA Intelligence Computing Architecture)
-----------------------------------------------

YICA is a revolutionary Compute-in-Memory (CIM) architecture designed specifically for AI computing optimization. By integrating computing units directly into memory, it significantly reduces data movement and provides exceptional performance and energy efficiency.

Key Features:

* **Compute-in-Memory Architecture**: 512 CIM arrays providing high parallel computing capability
* **Three-tier Memory Hierarchy**: Register file, SPM (Scratchpad Memory), and DRAM for optimized memory management
* **YIS Instruction Set**: Custom instruction set designed specifically for CIM architecture
* **Multi-backend Support**: Seamless switching between CUDA, Triton, and YICA backends
* **Automatic Optimization**: Intelligent search for optimal computation graphs
* **High Performance**: 2-3x performance improvement compared to traditional solutions

YiRage (AI Kernel Super Optimizer)
----------------------------------

YiRage is a next-generation AI kernel optimizer that combines YICA architecture awareness with intelligent optimization strategies. It serves as a code transformation and optimization tool designed to maximize performance across different hardware backends.

Core Capabilities:

* **Architecture-Aware Optimization**: Deep integration with YICA CIM architecture characteristics
* **Multi-objective Search**: Balances latency, energy efficiency, and memory utilization
* **Automatic Code Generation**: Transforms high-level models into optimized kernels
* **Cross-platform Support**: Works across different hardware environments
* **Hierarchical Optimization**: Multi-level optimization from algorithm to instruction level

Performance Metrics
===================

.. list-table:: Performance Comparison Table
   :widths: 25 25 25 25
   :header-rows: 1

   * - Operator Type
     - vs PyTorch
     - vs CUDA
     - vs Triton
   * - Matrix Multiplication
     - 3.0x
     - 2.2x
     - -
   * - Attention Mechanism
     - 2.8x
     - 1.9x
     - 1.5x
   * - End-to-End Inference
     - 2.5x
     - 1.7x
     - -

Related Links
=============

* `Source Code <../yirage/>`_ - YiRage core source code
* `Examples <../yirage/demo/>`_ - Usage examples and demonstrations
* `Test Suite <../tests/>`_ - Complete test cases

Getting Help
============

If you encounter issues while using the system, please:

1. Consult the relevant documentation
2. Check :doc:`FAQ <getting-started/quick-reference>`
3. View :doc:`Error Handling Guide <design/error_handling_logging_system>`
4. Submit an Issue or contact the maintenance team

.. raw:: html

   <div class="admonition note">
   <p class="admonition-title">Documentation Version</p>
   <p><strong>Version</strong>: v1.0.4<br>
   <strong>Last Updated</strong>: August 2025<br>
   <strong>Maintenance Team</strong>: YICA Development Team</p>
   </div>
