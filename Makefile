$(shell rm -f parsetab.py *.so *.pyc _*.py)
BENCHMARKS=$(wildcard *.py)
COMPILERS=python pythran parakeet pypy numba

# prevent any OpenMP based parallelism
export OMP_NUM_THREADS=1

# do not run the make command in parallel
.NOTPARALLEL:

setup=`grep -E '\#setup: ' $(1) | sed -e 's/\#setup: //'`
run=`grep -E '\#run: ' $(1) | sed -e 's/\#run: /res = /'` # note we are storing the result in a variable to avoid counting deallocation time

TARGETS=$(shell for compiler in $(COMPILERS); do for benchmark in $(BENCHMARKS:%.py=%); do printf "$$compiler-$$benchmark.timing " ; done ; done)

all:$(TARGETS)

# each rule below describes how to run a benchmark for a given compiler
# the ``timeit'' module is used to gather the results
python-%.timing:%.py
	@printf '$* python: '
	@python -m timeit -s "$(call setup,$<); from $* import $*" "$(call run, $<)"

pypysetup=`grep -E '\#setup: ' $(1) | sed -e 's/\#setup: //' -e 's/numpy/numpypy/g'`
pypy-%.timing:%.py
	@printf '$* pypy: '
	@pypy -m timeit -s "$(call pypysetup,$<); from $* import $*" "$(call run, $<)" || echo unsupported

pythran-%.timing:%.py
	@printf '$* pythran: '
	@(pythran $< && rm -f parsetab.py && python -m timeit -s "$(call setup,$<); from $* import $*" "$(call run, $<)" ) || echo unsupported
	@rm -f *.so

parakeet-%.timing:%.py
	@sed -e 's/def $*/import parakeet\n@parakeet.jit\ndef $*/' $< > _$<
	@printf '$* parakeet: '
	@python -m timeit -s "$(call setup,$<); from _$* import $*" "$(call run, $<)" || echo unsupported
	@rm -f _$<

numba-%.timing:%.py
	@sed -e 's/def /import numba\n@numba.autojit\ndef /' $< > _$<
	@printf '$* numba: '
	@python -m timeit -s "$(call setup,$<); from _$* import $*" "$(call run, $<)" || echo unsupported
	@rm -f _$<

clean:
	$(RM) callgrind.*
