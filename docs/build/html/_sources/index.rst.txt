=================================
Welcome to Minuet's Documentation
=================================

.. role:: strike
   :class: strike

Minuet🎶 is :strike:`a social dance for two people, usually in 3/4 time` a library that efficiently implements sparse convolutions (SC) for point clouds on GPUs.
Minuet use **sorted tables** and **binary search** for building kernel maps, which results in on average
:math:`15.8\times` (up to :math:`26.8\times`) speedup compared to hash table implementations in existing libraries.

.. toctree::
   :hidden:

   APIs <api>

Installation
============

.. code-block:: bash

   pip3 install "torch~=2.1" "packaging~=23.2"
   CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip3 install .

APIs
====

Please refer to :doc:`here <api>` for detailed APIs.

Examples
========

Coming soon.

Code Structure and Call Stack
=============================

Minuet is implemented as a torch extension. Every functionality of Minuet is
implemented with a Python interface with compiled C++ extension binary which
consists of three layers of function calls:

* PyTorch Wrappers: This layer interacts with the calls from Python and is
  responsible for calling the corresponding C++ wrappers.
* C++ Function Wrappers: We use a C++ function wrapper to avoid including the
  PyTorch's header to reduce the compile time during development. This layer is
  also responsible for launching CUDA kernels.
* CUDA Kernels: This layer implements the core functionality of Minuet on GPUs.

The following figure demonstrates the call stack of the kernel map building
process.

.. image:: _static/code-structure.svg
  :alt: The calling stack of the kernel map building process

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
