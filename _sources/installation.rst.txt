Installation
===========

Prerequisites
------------

* Python 3.8 or higher
* NumPy
* (Optional) CuPy for GPU support

Installation Methods
------------------

From PyPI
^^^^^^^^

For CPU-only support:

.. code-block:: bash

   pip install cortex_tensor

For GPU support:

.. code-block:: bash

   pip install cortex_tensor[gpu]

From Source
^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/yourusername/cortex.git
   cd cortex
   pip install -e .

Development Installation
^^^^^^^^^^^^^^^^^^^^^^

For development, install additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"