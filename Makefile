$(shell rm -f parsetab.py *.so *.pyc _*.py)
BENCHMARKS=$(wildcard *.py)
COMPILERS=python pythran parakeet

export OMP_NUM_THREADS=1

.NOTPARALLEL:

setup=`grep -E '\#setup: ' $(1) | sed -e 's/\#setup: //'`
run=`grep -E '\#run: ' $(1) | sed -e 's/\#run: //'`

all: $(COMPILERS:%=%.bench)


%.bench:
	$(MAKE) $(BENCHMARKS:%.py=$*-%.timing)

python-%.timing:%.py
	@printf '$* python: '
	@python -m timeit -s "$(call setup,$<); from $* import $*" "$(call run, $<)"

pythran-%.timing:%.py
	@pythran $<
	@printf '$* pythran: '
	@rm -f parsetab.py
	@python -m timeit -s "$(call setup,$<); from $* import $*" "$(call run, $<)"
	@rm -f *.so

parakeet-%.timing:%.py
	@sed -e 's/def $*/import parakeet\n@parakeet.jit\ndef $*/' $< > _$<
	@printf '$* parakeet: '
	@python -m timeit -s "$(call setup,$<); from _$* import $*" "$(call run, $<)" || echo nothing
	@rm -f _$<

clean:
	$(RM) callgrind.*
