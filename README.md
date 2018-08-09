# Implementation of HTM Spatial Pooler algorithm in CUDA

This is my CUDA implementation of [Numenta's Spatial Pooling algorithm](https://www.frontiersin.org/articles/10.3389/fncom.2017.00111/full).

My goal starting this project was to create an efficient CUDA backend for the entire [Numenta's Platform for Intelligent Computing](https://github.com/numenta/nupic) (NuPIC). I am open to starting an OSS project. If you are interested, do not hesitate to contact me!

## Details of CUDA implementation
Each SP column is mapped to a single thread. Unlike the original sequential version, inhibition is performed only per CUDA block of 1024 threads (some newer devices allow 2048 threads per block). This can be further improved in a future version, e.g. by mapping multiple columns to a single thread. This also means that the size of the spatial pooler is limited by the maximum number of threads your device can spin per kernel and the amount of global memory.

Initial permanences, potential connections and boost factors are initialized on the device - the only data transferred is the input vector. Transfer from pinned memory is faster, but incomparably slower to allocate. The potential connections are stored indeces with regards to the starting point to the block of input which belongs to corresponding CUDA block, and hence the overlap can be calculated simply as a block-wise CSR matrix-vector multiplication (with boost factors considered), input values being stored in shared memory. According to [this](http://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf) paper, this should be reasonably efficient choice. 

## Future optimization
Although most methods make extensive use of parallel, block-wise reduction to speed up their execution, some further opitmization can be made: `adaptSynapses()` exhibits some thread divergence (but I don't know how this can be evaded), and `inhibitColumns()` can potentially use more efficient sorting method (this probably does not present major bottleneck. however).

## Performance
This code performs significantly faster than the sequential C++ implementation, and can handle input of almost unlimited size. For example, with input of 131072 neurons and 32768 SP columns, the kernel performs one iteration in 28.536 ms on (rather obsolete) GeForce GTX 670, while handles input and number of columns of half the size (since it doesn't allow larger inputs) in 268.98 ms - 9.43x speedup on larger input.

## Requirements
* nVidia GPU with CUDA compatibility version >= 2.0 (most Tesla products and newer)
* CUDA Toolkit installed
* A C++ compiler (such as GCC)

You can check for CUDA compatibility of your device [here](https://developer.nvidia.com/cuda-gpus).
The newest version of CUDA toolkit is available for download [here](https://developer.nvidia.com/cuda-downloads).

## How to build
To build the main program, simply type

``
nvcc HelloSP.cu -o HelloSP -std=c++11
``

Building the unit tests is done analogously

``
nvcc UnitSP.cu -o UnitSP -std=c++11
``

## Unit tests
Unit tests are run simply by running `UnitSP.cu`
