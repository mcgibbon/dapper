******
dapper
******
----------------------------------------------------------------------
Dataset Analysis, Processing, and Presentation of Experimental Results
----------------------------------------------------------------------

**dapper** is a set of Python programming utilities written for the purpose
of analyzing and comparing observations taken during the MAGIC field campaign
against LES model runs. It is meant mainly for personal use, but can be
extended to work for other campaigns and models with similar data needs.

Features
========

* defined constants used in atmospheric science

* functions for common atmospheric science equations

* a simple calculate() interface function for accessing equations

* no need to remember equation function names or argument order

* fast calculation of quantities using numexpr

* skew-T plots integrated into matplotlib

Dependencies
============

I use this module on Python 2.7. I try to write in a way that will
work on Python 3.x, but no guaruntees.

Package dependencies:

* numpy

* six

* xarray

* matplotlib

Installation
============

To install this module, download and run the following:

.. code:: bash

    $ python setup.py install

If you would like to edit and develop the code, you can instead install in develop mode

.. code:: bash

    $ python setup.py develop

License
=======

This module is available under an MIT license. Please see ``LICENSE.txt``.

