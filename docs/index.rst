YICA/YiRage Documentation Center
================================

Welcome to the YICA (YICA Intelligence Computing Architecture) and YiRage (Super Optimization Engine) documentation center.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   getting-started/design-philosophy
   getting-started/quick-reference

.. toctree::
   :maxdepth: 2
   :caption: Architecture Design

   architecture/yica-architecture
   architecture/yirage-architecture
   architecture/modular-architecture
   architecture/implementation-summary
   architecture/mirage-integration-plan
   architecture/mirage-extension
   architecture/mirage-updates

.. toctree::
   :maxdepth: 2
   :caption: Development Guide

   development/performance-testing

.. toctree::
   :maxdepth: 2
   :caption: Deployment & Operations

   deployment/docker-deployment
   deployment/qemu-setup
   deployment/deployment-report

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/analyzer

.. toctree::
   :maxdepth: 2
   :caption: Production Design

   design/build_system_redesign
   design/compatibility_layer_enhancement
   design/configuration_management_system
   design/deployment_packaging_strategy
   design/error_handling_logging_system
   design/testing_framework_design

.. toctree::
   :maxdepth: 2
   :caption: Project Management

   project-management/backend-integration
   project-management/implementation-analysis
   project-management/roadmap
   project-management/execution-plan

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

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

YiRage (Super Optimization Engine)
---------------------------------

YiRage is a high-performance AI operator optimization engine that supports multiple backends (CUDA, Triton, YICA). It can automatically search and optimize computation graphs of AI models, achieving significant performance improvements.

Core Features
=============

* **Compute-in-Memory Architecture**: 512 CIM arrays for highly parallel computing
* **Three-Level Memory Hierarchy**: Optimized memory management with register files, SPM, and DRAM
* **YIS Instruction Set**: Instruction set specifically designed for CIM architecture
* **Multi-Backend Support**: Seamless switching between CUDA, Triton, and YICA backends
* **Automatic Optimization**: Intelligent search for optimal computation graphs
* **High Performance**: 2-3x performance improvement compared to traditional solutions

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
