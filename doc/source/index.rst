====================================================
Ladim: Lagrangian Advection and Diffusion Module
====================================================

Current version: |package_version|


What is Ladim?
===================

Ladim is an offline ocean particle tracking model written in Python, similar to
`OpenDrift <https://opendrift.github.io/>`_ and
`OceanParcels <https://oceanparcels.org/>`_. It was created by Bjørn Ådlandsvik
to support the research activity at the
`Institute of Marine Research <https://www.hi.no/>`_.

The first version of Ladim was written in Fortran in 1994 and used to study
`dispersal of cod eggs <https://doi.org/10.17895/ices.pub.19271159>`_
along the Norwegian Coast. Since its introduction, the software has been applied
in :doc:`numerous publications <publications>` and has undergone continuous
revision. The current version powers the on-demand management tool
`Norwegian Current Information System <https://stromkatalogen.hi.no/apps/ncis/v1/en/>`_ 
for particle dispersion along the Norwegian coast. It is also used for modelling
the
`dispersal of sea lice <https://www.hi.no/forskning/marine-data-forskningsdata/lakseluskart/html/lakseluskart.html>`_
in Norway, an important component of the
`Traffic Light System <https://trafikklyssystemet.no/>`_ 
that regulates salmon aquaculture in Norway.


.. _citation:

Citation
========

If you use the software in a publication or report, please cite it as follows:

Ådlandsvik, B., and Sundby, S. (1994). *Modelling the transport of cod larvae 
from the Lofoten area*. In ICES marine science symposia **198**,
`<https://doi.org/10.17895/ices.pub.19271159>`_


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

    algorithm
    config
    examples
    publications
    contributing
    credits
    autoapi/index
