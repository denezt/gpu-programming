

all: compile test
	@printf ""

compile:
	@figlet "Compiling Code"
	@../../bin/nvc-wrapper --action=compile --progname=unified_memory_test.cu

test:
	@figlet "Running Test"
	@./builds/unified_memory_test
