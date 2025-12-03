# Zero-copy mapped host memory 


### Description

Zero-copy mapped host memory (keep data in RAM, GPU reads it “remotely”)

``` text
This uses pinned + mapped host memory (cudaHostAllocMapped). The data physically lives in system RAM; the GPU accesses it over PCIe without an explicit cudaMemcpy. On integrated GPUs / UMA systems this is particularly useful.
```


``` text
nvcc -O2 zerocopy_host_mapped.cu -o zerocopy_host_mapped
./zerocopy_host_mapped
```

### Meaning:

* No explicit cudaMemcpy was used.
* Data stayed in system RAM (pinned).
* GPU kernel updated it by accessing that RAM over PCIe.
* This is the closest to “RAM holds data, GPU only computes” in a classic dGPU setup.


