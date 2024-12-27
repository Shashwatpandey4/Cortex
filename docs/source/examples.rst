Examples
========

Basic Operations
--------------

Creating and Manipulating Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import cortex as ctx
    
    # Create a tensor
    x = ctx.Tensor([1, 2, 3, 4, 5])
    
    # Basic operations
    y = x * 2
    z = x + y
    
    print(z)  # Tensor([3, 6, 9, 12, 15])

Automatic Differentiation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import cortex as ctx
    
    # Create input tensor
    x = ctx.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Forward pass
    y = x ** 2
    
    # Backward pass
    y.backward(ctx.Tensor([1.0, 1.0, 1.0]))
    
    print(x.grad)  # Prints gradients

GPU Acceleration
^^^^^^^^^^^^^

.. code-block:: python

    import cortex as ctx
    
    # Move computation to GPU if available
    if ctx.device.CUDA_AVAILABLE:
        ctx.set_default_device('cuda')
        
    # Create tensor (will be on GPU if available)
    x = ctx.Tensor([1, 2, 3])