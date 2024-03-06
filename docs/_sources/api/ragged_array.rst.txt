RaggedArray
----
A RaggedArray is a 2d array, where the lengths of the rows can differ. These arrays behaves as much as possible like numpy array, meaning that you can index it like a numpy array, call unfuncs on them and also call a limited set of array functions on them.


API documentation
===================
.. currentmodule:: npstructures
.. autoclass:: RaggedArray
   :members: 
.. currentmodule:: npstructures.arrayfunctions
.. autofunction:: concatenate
.. autofunction:: diff
.. autofunction:: zeros_like
.. autofunction:: ones_like
.. autofunction:: empty_like
.. autofunction:: where
.. autofunction:: unique
