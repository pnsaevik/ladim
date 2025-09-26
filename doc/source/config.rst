==============
Configuration
==============
   
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
Here we describe the different options available:


