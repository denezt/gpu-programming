# Unified Memory Test

### Description

Unified Memory oversubscription test (CPU RAM used transparently)

``` text
This uses cudaMallocManaged (Unified Memory). If you allocate more than your VRAM, the driver will spill into system 
RAM and page data over PCIe while the GPU computes.
```

``` bash
$ nvcc -O2 unified_memory_test.cu -o unified_memory_test
$ ./unified_memory_test  # default ~6 GiB
# or
$ ./unified_memory_test 10  # 10 GiB
```

### What this does:

1. Allocates more memory than VRAM using Unified Memory.
2. Initializes it on CPU (so it’s in system RAM).
3. GPU kernel scale_kernel runs over it.
4. Driver migrates pages back and forth between VRAM and RAM.

### You should see:

* It works even if it doesn't fit in VRAM.
* It becomes very slow once you exceed VRAM capacity.
* Use nvidia-smi dmon or nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv in another terminal to watch VRAM usage and activity.
