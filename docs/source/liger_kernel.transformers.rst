.. automodule:: liger_kernel.transformers
   :members:

Liger Cross Entropy Function
=============================

.. autoclass:: LigerCrossEntropyFunction
   :members:

   The Liger Cross Entropy Function is a custom autograd function that implements the Liger Cross Entropy loss.
   It overrides the forward and backward methods of the torch.autograd.Function class.

  .. automethod:: forward

  .. automethod:: backward

Liger Cross Entropy Forward
============================

.. autofunction:: cross_entropy_forward

Liger Cross Entropy Backward
============================

.. autofunction:: cross_entropy_backward

Liger Cross Entropy Kernel
==========================

.. autofunction:: liger_cross_entropy_kernel

Element Mul Kernel
===================

.. autofunction:: element_mul_kernel

