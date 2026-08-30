prog := $(PROG)

all: compile test
	@printf "\033[35mFinish building,\033[32m $(prog)\033[0m\n"
compile:
	@figlet "Compiling Code"
	@../bin/nvc-wrapper --action=compile --progname=$(prog).cu
test:
	@figlet "Running Test"
	@./builds/$(prog)
