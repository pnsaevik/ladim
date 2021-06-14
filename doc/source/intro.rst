Introduction
============

``LADiM`` is the acronym of the Lagrangian Advection and DIffusion Model.

``LADiM`` is a model for transport and spreading in the ocean. It uses a Lagrangian
particle tracking approach. The set of particles may typically represent the spatial
distribution of marine organisms or pollution. The particles may have individual
*behaviour* such as vertical movement by buoyancy or swimming and development like growth
or death. This is handled by a plug-in Individual Based Model (IBM). A particular focus
of ``LADiM`` is long simulations with many particles and complex biological behaviour.
The model is *off-line*, in the sense that it is forced by precomputed current fields
from an ocean circulation model.

``LADiM`` may be run as a flexible application governed by a configuration file and
plug-in modules. For further flexibility it is also available as a python library. It can
in principle use forcing from any ocean model through the plug-in system. Presently it
has modules for the Regional Ocean Model System, `ROMS <http://www.myroms.org>`_.

The `Institute of Marine Research <https://www.hi.no/en>`_ can trace a particle tracking
model called LADiM back to the 1990-ies. This old code was written in fortran. The
present LADiM model is implemented from scratch in the python programming language. It
has been actively developed and used since 2017 with version 2 released in 2021.

The model code is available under the `MIT license
<https://opensource.org/licenses/MIT>`_, and is hosted on github,
`https://github.com/bjornaa/ladim <https://github.com/bjornaa/ladim>`_. This
documentation is hosted on `Read the Docs
<https://ladim.readthedocs.io/en/master>`_. A `pdf version
<https://media.readthedocs.org/pdf/ladim/master/ladim.pdf>`_ is also available.

The flexibility of the python language makes it a natural choice for this kind of model.
There are alternative particle tracking models written in python. Most notable are `Ocean
Parcels <https://oceanparcels.org>`_ and `OpenDrift <https://opendrift.github.io/>`_. An
older model, `Paegan-Transport <https://github.com/asascience-open/Paegan-Transport>`_,
does not seem to be under active development anymore. There is considerable overlap in
functionality among these models. Python has a reputation for being slow. In practice
this is not a problem for ``LADiM``. For most applications the bottleneck is reading
large NetCDF files with model output. This is independent of language as the NetCDF
library written in C is used. ``LADiM`` uses the effective ``numpy`` library for
computations with a few functions using ``numba`` for an extra speed boost.

``LADiM`` is used extensively at the `Institute of Marine Research
<https://www.hi.no/en>`_. The traditional use for particle tracking is transport of fish
eggs and larvae. Presently a major subject is the study of spreading and abundance of
salmon lice. Operationally, `weekly maps
<https://www.hi.no/forskning/marine-data-forskningsdata/lakseluskart/html/lakseluskart.html>`_
of salmon lice copepodites are made available. Results from LADiM simulations are also
crucial in the "traffic light" system for management of the salmon aquaculture industry
in Norway. [Find good reference in English]. The Norwegian Current Information System,
`NCIS <https://ncis.imr.no>`_, also uses ``LADiM`` to produce spreading maps.

There are several scientific articles using ``LADiM`` output. A list is given below.
Please notify Bjørn Ådlandsvik <bjorn@hi.no> on new papers, so the list can be kept up to
date. LADiM may be cited as ....

.. toctree::

    Scientific articles using LADiM <articles.rst>