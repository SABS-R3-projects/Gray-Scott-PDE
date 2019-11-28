.. Gray-Scott PDE documentation master file, created by
   sphinx-quickstart on Tue Nov 26 22:41:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gray-Scott partial differential equation system
===============================================

**Augustin, Neiro, van der Plas**

The Gray-Scott PDE is a an example of a reaction diffusion system, i.e.
a systems of PDEs that combine linear diffusion with nonlinear interactions. Gray and Scott
formulated their model in 1983 as follows

:math:`u_t = \epsilon_1 \Delta u - u v^2 + F(1 - u)`

:math:`v_t = \epsilon_2 \Delta v + u v^2 - (k+F)v`

.. automodule:: GrayScottPDE
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
