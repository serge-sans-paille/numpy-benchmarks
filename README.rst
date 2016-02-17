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
pretty-print the result in various format (see ``fmt-bench --help``::

    python run.py | ./fmt-bench


Example
=======

    $  python run.py -t python benchmarks/harris.py
    harris Python 5431 5454 14

What does it mean? The code from ``benchmarks/harris.py`` was run through
``timeit`` using the ``#setup`` and ``#run`` code. It outputs (in that order
and in nanoseconds):

1. the **best** execution time among all runs;
2. the **average** execution time of the runs;
3. the **standard deviation** of the runs.
