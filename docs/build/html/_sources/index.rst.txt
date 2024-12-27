Welcome to Cortex's documentation!
================================

Cortex is a deep learning library with CPU and GPU support using NumPy and CuPy.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   tutorials/index
   api/index
   examples
   contributing

Quick Start
----------

.. code-block:: python

   import cortex as ctx
   
   # Create a tensor
   x = ctx.Tensor([1, 2, 3])
   
   # Move to GPU if available
   if ctx.device.CUDA_AVAILABLE:
       x = x.to('cuda')

Installation
-----------

CPU-only installation:

.. code-block:: bash

   pip install cortex_tensor

With GPU support:

.. code-block:: bash

   pip install cortex_tensor[gpu]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`