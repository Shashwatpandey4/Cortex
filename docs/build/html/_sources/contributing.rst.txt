Contributing
===========

We welcome contributions to Cortex! This document will guide you through the process.

Development Setup
---------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/yourusername/cortex.git
      cd cortex

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Running Tests
-----------

Run the test suite:

.. code-block:: bash

    pytest tests/

Code Style
---------

We follow PEP 8 guidelines. Please ensure your code is formatted using black:

.. code-block:: bash

    black cortex tests

Pull Request Process
-----------------

1. Create a new branch for your feature
2. Write tests for new functionality
3. Update documentation as needed
4. Submit a pull request
5. Wait for review and address any feedback

Documentation
-----------

Build the documentation locally:

.. code-block:: bash

    cd docs
    make html