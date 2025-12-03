

all: compile test
	@printf ""

compile:
	@figlet "Compiling Code"
	@../../bin/nvc-wrapper --action=compile --progname=zerocopy_host_mapped.cu

test:
	@figlet "Running Test"
	@./builds/zerocopy_host_mapped