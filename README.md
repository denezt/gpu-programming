# GPU PROGRAMMING

### Sessions and Examples for Faster Computational processing

``` sh
$./bin/nvc-wrapper --help
```

### Compile the program

``` sh
$./bin/nvc-wrapper --compile progname:NAME_OF_SOURCE
```

### Clean and Remove build sessions

``` sh
$./bin/nvc-wrapper --flush-all
```

### Alternative calls

``` sh
$./bin/nvc-wrapper --action=compile progname:NAME_OF_SOURCE
```

### Making changes to the NVC Wrapper
1. Open the `nvc_wrapper` script and edit the following line to match the actual location of you `nvcc` (Nvidia CUDA Compiler).

``` sh
export PATH=$PATH:/usr/local/cuda-12.2/bin
```

2. Below is an example of how we update to **cuda-12.4**.

``` sh
export PATH=$PATH:/usr/local/cuda-12.4/bin
```
