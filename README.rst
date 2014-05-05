================
Numpy Benchmarks
================

A collection of scientific kernels that use the numpy package, for benchmarking
purpose.

Each kernel holds a ``#setup: ... code ...`` and a ``#run: ... code ...``
comment line to be passed to the ``timeit`` module for easy benchmarking, as
automated by the _clumsy_ Makefile.

To run a specific set of benchmarks on a specific set of compilers, use the
eponymous Makefile variables, as in::

    make COMPILERS=python\ pythran BENCHMARKS=harris.py\ evolve.py

