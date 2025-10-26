PRG1=gradient_algo
NVC=/usr/local/cuda-12.4/bin/nvcc

all: clean compile test
	@printf "Completed build for $(PRG1)\n"

clean:
	find . -type f -name $(PRG1) -delete

compile:
	@$(NVC) $(PRG1).cu -o $(PRG1)

test:
	@./$(PRG1)
