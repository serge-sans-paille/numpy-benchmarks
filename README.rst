================
Numpy Benchmarks
================

A collection of scientific kernels that use the numpy package, for benchmarking
purpose.

Each kernel holds a ``#setup: ... code ...`` and a ``#run: ... code ...``
comment line to be passed to the ``timeit`` module for easy benchmarking, as
automated by the ``run.py`` script.
To run a specific set of benchmarks on a specific set of compilers, use the
ad hoc arguments , as in::

    python run.py -t python -t pythran benchmarks/harris.py benchmarks/evolve.py

A small utility, provided with the benchmark, ``fmt-bench`` can be used to
pretty-print the result::

    python run.py | ./fmt-bench
