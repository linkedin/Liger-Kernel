Liger Kernel API
================

.. automodule:: liger_kernel
   :members:

Submodules
----------
.. toctree::
   :maxdepth: 2

   liger_kernel.transformers
   liger_kernel.nn

Transformers Module
-------------------

The `liger_kernel.transformers` module includes essential tools and model classes designed for tasks such as language modeling and loss calculations, optimizing deep learning workflows.

.. automodule:: liger_kernel.transformers
   :members:

Classes
-------

AutoLigerKernelForCausalLM
--------------------------

The `AutoLigerKernelForCausalLM` class is an extension for causal language modeling, providing enhanced support for auto-regressive tasks.

.. autoclass:: liger_kernel.transformers.AutoLigerKernelForCausalLM
   :members:
   :undoc-members:

LigerFusedLinearCrossEntropyLoss
--------------------------------

Implements a fused linear layer with Cross-Entropy Loss, offering computational efficiency improvements.

.. autoclass:: liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss
   :members:
   :undoc-members:

Loss Functions
--------------

KLDivergence
------------

The `KLDivergence` class offers a divergence metric based on Kullback-Leibler Divergence, commonly used in various ML training contexts.

.. autoclass:: liger_kernel.transformers.KLDivergence
   :members:
   :undoc-members:

JSD
---

This class defines the Jensen-Shannon Divergence (JSD), used for calculating symmetrical divergence between distributions.

.. autoclass:: liger_kernel.transformers.JSD
   :members:
   :undoc-members:

GeneralizedJSD
--------------

Generalized form of Jensen-Shannon Divergence, adapted for more complex applications.

.. autoclass:: liger_kernel.transformers.GeneralizedJSD
   :members:
   :undoc-members:

FusedLinearJSD
--------------

Provides a fused linear implementation of Jensen-Shannon Divergence for faster computation.

.. autoclass:: liger_kernel.transformers.FusedLinearJSD
   :members:
   :undoc-members:

Experimental Kernels
---------------------

A collection of experimental kernels for advanced and experimental features in `liger_kernel`.

.. automodule:: liger_kernel.transformers.experimental
   :members:

Neural Network Module
---------------------

The `liger_kernel.nn` module offers fundamental neural network building blocks.

.. automodule:: liger_kernel.nn
   :members:

Module Class
------------

The base `nn.Module` class provides essential methods and attributes for building neural network models.

.. autoclass:: liger_kernel.nn.Module
   :members:
   :undoc-members:
