====================================================
Ladim: Lagrangian Advection and Diffusion Module
====================================================

Current version: |package_version|


What is Ladim?
===================

This is ladim.

.. _citation:

Citation
========

This is how you cite.


Installation
============

The package is installed using pip:

::

  pip install ladim

If you use Ladim on Windows, consider installing the prerequisites using
conda-forge instead:

::

  conda install -c conda-forge netCDF4 numba numexpr numpy pandas pyarrow pyproj pyyaml scipy xarray


Usage
=====

The software can be started from the command line as

.. code-block::

    ladim ladim.yaml

or from within python as

.. code-block:: python

    import ladim
    ladim.run("ladim.yaml")

In both cases, simulation details are specified in the
file ``ladim.yaml``, written in the `YAML file format <https://yaml.org/spec/>`_.
The :ref:`examples_page` section includes many examples of valid config files,
for various types of problems.


Documentation
=============
.. toctree::
    :maxdepth: 2

    examples
