PRG1=gpu_vector_threads
NVC=/usr/local/cuda-12.4/bin/nvcc

all: clean compile test
	@printf "Completed build for $(PRG1)\n"

clean:
	find . -type f -name $(PRG1) -delete
	find . -type f -name "gpu_vector.txt" -delete
	find . -type f -name "*.log" -delete

compile:
	@$(NVC) $(PRG1).cu -o $(PRG1)

test:
	@./$(PRG1)
