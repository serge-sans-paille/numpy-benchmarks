'''
An adaptation from timeit that outputs some extra statistical informations
'''

from timeit import default_timer, default_repeat, Timer
import numpy
import sys


def main(args=None):
    """Main program, used when run as a script.

    The optional argument specifies the command line to be parsed,
    defaulting to sys.argv[1:].

    The return value is an exit code to be passed to sys.exit(); it
    may be None to indicate success.

    When an exception happens during timing, a traceback is printed to
    stderr and the return value is 1.  Exceptions at other times
    (including the template compilation) are not caught.
    """
    if args is None:
        args = sys.argv[1:]
    import getopt
    try:
        opts, args = getopt.getopt(args, "n:s:r:tcvh",
                                   ["number=", "setup=", "repeat=",
                                    "time", "clock", "verbose", "help"])
    except getopt.error, err:
        print err
        print "use -h/--help for command line help"
        return 2
    timer = default_timer
    stmt = "\n".join(args) or "pass"
    number = 0  # auto-determine
    setup = []
    repeat = default_repeat
    verbose = 0
    precision = 3
    for o, a in opts:
        if o in ("-n", "--number"):
            number = int(a)
        if o in ("-s", "--setup"):
            setup.append(a)
        if o in ("-r", "--repeat"):
            repeat = int(a)
            if repeat <= 0:
                repeat = 1
        if o in ("-t", "--time"):
            timer = time.time
        if o in ("-c", "--clock"):
            timer = time.clock
        if o in ("-v", "--verbose"):
            if verbose:
                precision += 1
            verbose += 1
        if o in ("-h", "--help"):
            print __doc__,
            return 0
    setup = "\n".join(setup) or "pass"
    # Include the current directory, so that local imports work (sys.path
    # contains the directory of this script, rather than the current
    # directory)
    import os
    sys.path.insert(0, os.curdir)
    t = Timer(stmt, setup, timer)
    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        for i in range(1, 10):
            number = 10**i
            try:
                x = t.timeit(number)
            except:
                t.print_exc()
                return 1
            if verbose:
                print "%d loops -> %.*g secs" % (number, precision, x)
            if x >= 0.2:
                break
    try:
        r = t.repeat(repeat, number)
    except:
        t.print_exc()
        return 1
    if verbose:
        print "raw times:", " ".join(["%.*g" % (precision, x) for x in r])
    r = [int(x * 1e6 / number) for x in r]
    best = min(r)
    average = int(numpy.average(r))
    std = int(numpy.std(r))

    print best, average, std

if __name__ == "__main__":
    sys.exit(main())
