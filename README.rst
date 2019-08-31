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

Install
=======

Through pip::

    > pip install numpy_benchmarks

Or locally::

    > python setup.py install

Usage
=====

To list available benchmarks::

    > np-bench list

To run the whole benchmark suite and save the output::

    > np-bench run -o run.log


To post-process the output of ``np-bench run``, for
instance to plot the result as ``png``::

    > np-bench format -tpng run.log

To compare multiple version of the same tool, the following can be handy::

    > np-bench run -tpythran -o ref.log -p ref-
    > # change pythran version, branch, whatever
    > np-bench run -tpythran -o new.log -p new-
    > np-bench format ref.log new.log

=======
    > np-bench run -tpython ${prefix}/benchmarks/harris.py
    harris Python 5431 5454 14

What does it mean? ``python`` is used as single engine through ``-tpython``. The
code from ``${prefix}/benchmarks/harris.py`` (where you can get the value of
``${prefix}`` from ``np-bench list``) was run through ``timeit`` using the
``#setup`` and ``#run`` code. It outputs (in that order and in nanoseconds):

1. the **best** execution time among all runs;
2. the **average** execution time of the runs;
3. the **standard deviation** of the runs.
