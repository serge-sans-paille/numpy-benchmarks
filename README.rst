================
Numpy Benchmarks
================

A collection of scientific kernels that use the numpy package, for benchmarking
purpose.

Usage
=====

First setup the benchmarking environment::

    > ./np-bench setup

Then run the whole benchmark suite::

    > ./np-bench run

To run a specific set of benchmarks on a specific set of compilers, use the
ad hoc arguments , as in::

    > ./np-bench run -tnumba -tpythran benchmarks/harris.py benchmarks/evolve.py

It is possible to post-process the raw output of ``./np-bench run``, for
instance to plot the result as ``png``::

    > ./np-bench run -tpython -tpythran > run.log
    > ./np-bench format  -tpng run.log

Kernels
=======

Each kernel holds a ``#setup: ... code ...`` and a ``#run: ... code ...``
comment line to be passed to the ``timeit`` module for easy benchmarking, as
automated by the ``np-bench`` script.

Each kernel involve some high-level numpy construct, sometimes mixed with
explicit iteration.

Example
=======

Let's analyze some output::

    > ./np-bench run -tpython benchmarks/harris.py
    harris Python 5431 5454 14

What does it mean? The code from ``benchmarks/harris.py`` was run through
``timeit`` using the ``#setup`` and ``#run`` code. It outputs (in that order
and in nanoseconds):

1. the **best** execution time among all runs;
2. the **average** execution time of the runs;
3. the **standard deviation** of the runs.

:x
