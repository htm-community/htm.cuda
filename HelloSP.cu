#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <assert.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>

#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>

#include "SpatialPooler.cu"

#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace std;

typedef unsigned int UInt;
typedef float Real;


UInt* generatePotentialPoolsUsingShuffle(UInt* potentialPools, const UInt SP_SIZE, const UInt IN_BLOCK_SIZE, const UInt MAX_CONNECTED) 
{
	
	vector<UInt> indeces(IN_BLOCK_SIZE);
	iota(indeces.begin(), indeces.end(), 0);

	// We could also do this on the device
	// thrust::host_vector<UInt> indeces(IN_BLOCK_SIZE);
	// thrust::sequence(input_indeces.begin(), input_indeces.end(), 0, 1);

	for(int i=0; i < SP_SIZE; i++) {
		random_shuffle(indeces.begin(), indeces.end());
		copy(indeces.begin(), indeces.begin()+MAX_CONNECTED, &potentialPools[i*MAX_CONNECTED]);
		// This may slightly improve performance, but slows down initialization
		sort(&potentialPools[i*MAX_CONNECTED], &potentialPools[(i+1)*MAX_CONNECTED]);
	}

	return potentialPools;
}

bool* generate01(bool* ar, size_t size, Real inDensity)
{
	for(int i=0; i < size; i++)
	{
		ar[i] = (Real)(rand()%100)/100 <= inDensity ? 1 : 0;
	}
	return ar;
}

struct prg : public thrust::unary_function<unsigned int,bool>
{
	Real IN_DENSITY;

	__host__ __device__
		prg(Real ind) : IN_DENSITY(ind) {}
	
    __host__ __device__
        bool operator()(const unsigned int thread_id) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(0, 1);
            rng.discard(thread_id);

            return dist(rng) <= IN_DENSITY ? true : false;
        }
};

void visualize_input_generated_on_device(thrust::device_vector<bool>& in_vector, UInt* pot_pools_host, const UInt MAX_CONNECTED, const UInt SP_SIZE)
{
	printf("INPUT\n");
	thrust::copy(in_vector.begin(), in_vector.end(), std::ostream_iterator<bool>(std::cout, " "));
	printf("\n");
	// This overlows stdout buffer (better write to a file if necessary)
	// printf("POTENTIAL POOLS");
	// for(int i=0; i<SP_SIZE; i++)
	// {
	// 	for(int j=0; j<MAX_CONNECTED; j++)
	// 		printf("%d \t", pot_pools_host[i*MAX_CONNECTED+j]);
	// 	printf("\n");
	// }


}

void visualize_output(bool* cols_host, const UInt SP_SIZE, UInt BLOCK_SIZE)
{
	printf("OUTPUT\n");
	for(int i=0; i<SP_SIZE; i++)
	{
		printf("%d ", cols_host[i]);
		if(i % BLOCK_SIZE == 0 && i > 0)
			printf("\n");
	}
	printf("\n");
	
	// The final sparsity will approach target with increasing block size
	int ones = 0;
	for(int i=0; i < SP_SIZE; i++)
		if(cols_host[i] > 0) ones++;
	printf("Sparsity: %f \n", (Real)ones/SP_SIZE);

}

int main(int argc, const char * argv[])
{
	srand(time(NULL));
	
    // construct input args
    args ar;
	ar.iteration_num=0;
	ar.learn=true;
	ar.localAreaDensity=0.02; // SP density after inhibition
    ar.potentialPct=0.1; // 
    ar.connectedPct=0.5;
    ar.stimulusThreshold=0;
    ar.synPermTrimThreshold=0.025;
    ar.synPermMax=1.0;
    ar.synPermConnected=0.1;
	ar.synPermActiveInc=0.05;
	ar.synPermInactiveDec=0.008;
	ar.synPermBelowStimulusInc=ar.synPermConnected / 10.0;
	ar.dutyCyclePeriod=1000;
	ar.boostStrength=0.05; // 0 means no boosting
	ar.minPctOdc=0.001;
	ar.update_period=50;
	
	ar.SP_SIZE = 32768;
	ar.IN_SIZE = 131072;
	ar.BLOCK_SIZE = 1024;
	ar.NUM_BLOCKS = ar.SP_SIZE/ar.BLOCK_SIZE;
	ar.IN_BLOCK_SIZE = ar.IN_SIZE/ar.NUM_BLOCKS; // Size of chunk of input processed by a single cuda block
	ar.MAX_CONNECTED = 1024;
	ar.IN_DENSITY = 0.5; // Density of input connections
	ar.num_connected = std::floor(ar.MAX_CONNECTED*ar.connectedPct);

	// Host memory allocation
    bool* cols_host = (bool*) malloc(ar.SP_SIZE*sizeof(bool));
	UInt* pot_pools_host = (UInt*) malloc(ar.SP_SIZE*ar.num_connected*sizeof(UInt));
 	pot_pools_host = generatePotentialPoolsUsingShuffle(pot_pools_host, ar.SP_SIZE, ar.IN_BLOCK_SIZE, ar.num_connected);

	// Global memory pointers
	args* ar_dev;

	// Global memory allocation
    checkError( cudaMalloc((void **) &ar_dev, sizeof(ar)) );

	size_t pot_dev_pitch_in_bytes, per_dev_pitch_in_bytes;
	checkError( cudaMallocPitch((void **) &ar.pot_dev, &pot_dev_pitch_in_bytes, ar.num_connected*sizeof(UInt), ar.SP_SIZE) );
	checkError( cudaMallocPitch((void **) &ar.per_dev, &per_dev_pitch_in_bytes, ar.num_connected*sizeof(Real), ar.SP_SIZE) );
	ar.pot_dev_pitch = pot_dev_pitch_in_bytes / sizeof(UInt);
	ar.per_dev_pitch = per_dev_pitch_in_bytes / sizeof(Real);

	checkError( cudaMalloc((void **) &ar.boosts_dev, ar.SP_SIZE*ar.num_connected*sizeof(Real)) );
    checkError( cudaMalloc((void **) &ar.olaps_dev, ar.SP_SIZE*sizeof(UInt)) );
    checkError( cudaMalloc((void **) &ar.cols_dev, ar.SP_SIZE*sizeof(bool)) );
	checkError( cudaMalloc((void **) &ar.numPot_dev, ar.SP_SIZE*sizeof(UInt)) );
    checkError( cudaMalloc((void **) &ar.odc_dev, ar.MAX_CONNECTED*ar.SP_SIZE*sizeof(Real)) );
    checkError( cudaMalloc((void **) &ar.adc_dev, ar.MAX_CONNECTED*ar.SP_SIZE*sizeof(Real)) );
	checkError( cudaMalloc((void **) &ar.minOdc_dev, ar.NUM_BLOCKS*sizeof(Real)) );
	checkError( cudaMalloc((void **) &ar.dev_states, ar.SP_SIZE*ar.BLOCK_SIZE*sizeof(curandState)) );

	// Global memory initialization

	setup_kernel<<<ar.NUM_BLOCKS, ar.BLOCK_SIZE>>>(ar.dev_states);

	// Permanences
	generatePermanences<<<ar.SP_SIZE, ar.num_connected>>>(ar.per_dev, ar.per_dev_pitch, ar.connectedPct, ar.synPermConnected, ar.synPermMax, ar.dev_states);

	// Boosts
	thrust::device_ptr<float> boosts_ptr(ar.boosts_dev);
	thrust::fill(boosts_ptr, boosts_ptr+ar.SP_SIZE*ar.num_connected*sizeof(Real), 1.0);
	
	// Number of potentialy connected synapses - unnecessary if we want it variable
	thrust::device_ptr<UInt> num_ptr(ar.numPot_dev);
	thrust::fill(num_ptr, num_ptr+ar.SP_SIZE*sizeof(UInt), ar.num_connected);

	// Input
	thrust::device_vector<bool> in_vector(ar.IN_SIZE);

	thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin,
            index_sequence_begin + ar.IN_SIZE,
            in_vector.begin(),
            prg(ar.IN_DENSITY));

	ar.in_dev = thrust::raw_pointer_cast(&in_vector[0]);

	visualize_input_generated_on_device(in_vector, pot_pools_host, ar.num_connected, ar.SP_SIZE);

	// Memcpy to device
    checkError( cudaMemcpy(ar_dev, (void**) &ar, sizeof(ar), cudaMemcpyHostToDevice) );
    checkError( cudaMemcpy2D(ar.pot_dev, pot_dev_pitch_in_bytes, pot_pools_host, ar.num_connected*sizeof(UInt), ar.num_connected*sizeof(UInt), ar.SP_SIZE, cudaMemcpyHostToDevice) );

	// Kernel call
	size_t sm = ar.BLOCK_SIZE*(2*sizeof(Real) + sizeof(UInt)) + ar.IN_BLOCK_SIZE*sizeof(bool);
    compute<<<ar.NUM_BLOCKS, ar.BLOCK_SIZE, sm>>>(ar_dev);
    
	// Memcpy from device
    checkError( cudaMemcpy(cols_host, ar.cols_dev, ar.SP_SIZE*sizeof(bool), cudaMemcpyDeviceToHost)); 

	visualize_output(cols_host, ar.SP_SIZE, ar.BLOCK_SIZE);

	cudaFree(ar.cols_dev); 
	cudaFree(ar.pot_dev);
   	cudaFree(ar.per_dev); 
	cudaFree(ar.boosts_dev);
	cudaFree(ar.odc_dev); 
	cudaFree(ar.adc_dev); 
	cudaFree(ar.numPot_dev);
    
	return 0;
}
