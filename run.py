import re
import tempfile
import os.path
import random
import stat


class PythonExtractor(object):

    name = 'Python'

    def __init__(self, output_dir):
        self.re_setup = re.compile('^#setup: (.*)$')
        self.re_run = re.compile('^#run: (.*)$')
        self.output_dir = output_dir

    def process_lines(self, filename, lines):
        content = []
        for line in lines:
            m = self.re_setup.match(line)
            if m:
                setup = m.group(1)
            m = self.re_run.match(line)
            if m:
                run = 'res = ' + m.group(1)
            content.append(line)
        try:
            return setup, run, content
        except NameError as n:
            raise RuntimeError('%s has invalid header' % filename)

    def __call__(self, filename):
        with open(filename) as fd:
            s, r, c = self.process_lines(filename, fd)
            return s, r, ''.join(c)

    def compile(self, filename):
        pass


class PypyExtractor(PythonExtractor):

    name = 'pypy' 

class XtensorExtractor(PythonExtractor):

    name = 'xtensor'

    def __init__(self, output_dir):
        super(XtensorExtractor, self).__init__(output_dir)

    def process_lines(self, filename, lines):
        s, r, c = super(XtensorExtractor, self).process_lines(filename, lines)
        c = ['from xbench import {}'.format(os.path.basename(filename)[:-3])]
        return s, r, c

class PythranExtractor(PythonExtractor):

    name = 'pythran'

    def compile(self, filename):
        import pythran, os
        os.chdir(self.output_dir)
        try:
            pythran.compile_pythranfile(os.path.join('..', filename))
        finally:
            os.chdir('..')



class ParakeetExtractor(PythonExtractor):

    name = 'parakeet'

    def __init__(self, output_dir):
        super(ParakeetExtractor, self).__init__(output_dir)
        self.extra_import = 'import parakeet\n'
        self.decorator = '@parakeet.jit\n'

    def process_lines(self, filename, lines):
        s, r, c = super(ParakeetExtractor, self).process_lines(filename, lines)
        lines = [self.extra_import]
        for line in c:
            if line.startswith('def '):
                lines.append(self.decorator)
            lines.append(line)
        return s, r, lines


class NumbaExtractor(ParakeetExtractor):

    name = 'numba'

    def __init__(self, output_dir):
        super(NumbaExtractor, self).__init__(output_dir)
        self.extra_import = 'import numba\n'
        self.decorator = '@numba.jit\n'


class HopeExtractor(ParakeetExtractor):

    name = 'hope'

    def __init__(self, output_dir):
        super(HopeExtractor, self).__init__(output_dir)
        self.extra_import = 'import hope\n'
        self.decorator = '@hope.jit\n'


def run(filenames, extractors):
    location = tempfile.mkdtemp(prefix='rundir_', dir='.')
    shelllines = []
    for filename in filenames:
        for extractor in extractors:
            e = extractor(location)
            basename = os.path.basename(filename)
            function, _ = os.path.splitext(basename)
            tmpfilename = '_'.join([extractor.name, basename])
            tmpmodule, _ = os.path.splitext(tmpfilename)
            where = os.path.join(location, tmpfilename)
            try:
                setup, run, content = e(filename)
                open(where, 'w').write(content)
                e.compile(where)
                shelllines.append('printf "{function} {extractor} " && PYTHONPATH=..:$PYTHONPATH python -m benchit -r 11 -n 40 -s "{setup}; from {module} import {function} ; {run}" "{run}" 2>/dev/null || echo fail'.format(setup=setup, module=tmpmodule, function=function, run=run, extractor=extractor.name))
            except:
                shelllines.append('echo "{function} {extractor} unsupported"'.format(function=function, extractor=extractor.name))

    # random.shuffle(shelllines)
    shelllines = ['#!/bin/sh', 'cp ./xbench/xbench.so ./{}'.format(location), 'export OMP_NUM_THREADS=1', 'cd `dirname $0`'] + shelllines

    shellscript = os.path.join(location, 'run.sh')
    open(shellscript, 'w').write('\n'.join(shelllines))
    os.chmod(shellscript, stat.S_IXUSR | stat.S_IRUSR)
    return shellscript

if __name__ == '__main__':
    import glob
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog='numpy-benchmarks',
                                     description='run synthetic numpy benchmarks',
                                     epilog="It's a megablast!")
    parser.add_argument('benchmarks', nargs='*',
                        help='benchmark to run, default is benchmarks/*',
                        default=glob.glob('benchmarks/*.py'))
    default_targets=['xtensor', 'python', 'pythran', 'parakeet', 'numba', 'pypy', 'hope']
    # default_targets=['xtensor']
    parser.add_argument('-t', action='append', dest='targets', metavar='TARGET',
                        help='target compilers to use, default is %s' % ', '.join(default_targets))
    args = parser.parse_args(sys.argv[1:])

    if args.targets is None:
        args.targets = default_targets

    conv = lambda t: globals()[t.capitalize() + 'Extractor']
    args.targets = [conv(t) for t in args.targets]
    print(args.benchmarks)
    script = run(args.benchmarks, args.targets)
    os.execl(script, script)
