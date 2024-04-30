Welcome to Numpy Structures's documentation!
============================================

NpStructures provides some convenience classes that follows the numpy dispacth protocols (i.e). These classes are implemented using only NumPy as a dependency so can be included in any project that already uses Numpy.

To install

.. code-block:: bash

    pip install npstructures


.. _what_can_you_do:

What can you do with BioNumpy?
----------------------------------

.. grid:: 2

    .. grid-item-card:: :material-regular:`sort;3em`
        :text-align: center
        :link: api/ragged_array.html

        **RaggedArray**

        Work with arrays with unequal rowlength as NumPy arrays

    .. grid-item-card::  :material-regular:`tune;3em`
        :text-align: center
        :link: api/runlength_array.html

        **RunLengthArray**

        Smart representation of arrays with long streches of equal values

.. grid:: 2

    .. grid-item-card::  :material-regular:`calculate;3em`
        :text-align: center
        :link: api/bitarray.html

        **BitArray**

        Work with lower than 8bit representation of data

    .. grid-item-card:: :material-regular:`grid_on;3em`
        :text-align: center
        :link: api/hash_table.html

        **HashTable**

        Simple static Hash Table for numpy arrays
