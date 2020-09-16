#include <chrono>
#include <thread>
#include <iostream>
#include <random>
#include <cmath>
#include <atomic>

#include <stdio.h>

#include "Timer.cuh"
#include "CheckError.cuh"

#include <omp.h>

using namespace timer;

// Set PRINT to 1 for debug output
#define PRINT 1
#define FROM_debug 0
#define TO_debug 16

// Set ZERO to 1 to use Zero copy, set ZERO to 0 to use Unified Memory
#define ZERO 1

unsigned int N = 2;
const int POW = 16;
const float MINUTES = 0.1; // Dictates the length of the benchmark, but doesn't actually follow the length 
const int SUMS = 8;
const int BLOCK_SIZE_X = 512;
const int BLOCK_SIZE_Y = 1;


__global__
void gpu_compute(int* matrix, const int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    double fp0 = 2.0;
    double fp1 = 2.0;
	int mat = matrix[row];
	double res;
	for (int j = 0; j < 3; j++ ) {
		fp0 *= float(j) + atan(tgamma(sqrt(acosh(__ddiv_ru(3.14159265359 * mat, 0.7)))));
		fp1 += float(j) + tgamma(sqrt(acosh(__ddiv_ru(3.14159265359 * mat, 0.7))));
		fp0 *= float(j) * atan(tgamma(sqrt(acosh(__ddiv_ru(3.14159265359 * fp1, 0.7)))));
		fp1 *= float(j) / sqrt(tgamma(sqrt(acosh(__ddiv_ru(3.14159265359 * fp0, 0.7)))));
		res /= sqrt(fp0 + fp1);
	}
	if (17 % mat == 0) matrix[row] = res; // mat cannot be 17 or 1, so this statement always evaluates to false, forcing the compiler to actually execute the code in the for cycle (and not optimize it away)
}


void fill_data(int * d_matrix_host, int N){
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);
    
    for (int i = 0; i < N; i++) {
		int temp = distribution(generator);
		if (temp == 17 || temp == 1) temp++;
		d_matrix_host[i] = temp;
    }
}

int main() {
    N = (unsigned int) pow(N, POW);
    int grid = N / BLOCK_SIZE_X;
    // -------------------------------------------------------------------------
    // DEVICE INIT
    dim3 DimGrid(grid, 1, 1);
    if (N % grid) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    // -------------------------------------------------------------------------
    cudaSetDeviceFlags(cudaDeviceMapHost);
	
	Timer<HOST> TM;
	Timer<HOST> TM_update;
	Timer<HOST> TM_app;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int * d_matrix_host;
    int * d_matrix;    

    #if ZERO
    // Zero Copy Allocation
	SAFE_CALL(cudaHostAlloc((void **)&d_matrix_host, N * sizeof(int), cudaHostAllocMapped));
    SAFE_CALL(cudaHostGetDevicePointer((void **)&d_matrix, (void *) d_matrix_host , 0));
    #else
    // Unified Allocation    
  	SAFE_CALL(cudaMallocManaged((void **)&d_matrix_host, N * sizeof(int)));
  	#endif    
    
    // -------------------------------------------------------------------------
    // MATRIX INITILIZATION
    std::cout << "Starting Initialization..." << std::endl;
	TM.start();
    fill_data(d_matrix_host, N);
    TM.stop();
    TM.print("Initialization Finished, time: ");

    // -------------------------------------------------------------------------
    // EXECUTION
    TM_app.start();
    std::cout << "Starting computation (GPU+CPU)..." << std::endl;
	for (int i = 0; i < int((MINUTES*60*1000)/33.3); i++) {
		TM.start();
	    gpu_compute << < DimGrid, DimBlock >> > (d_matrix_host, N);
	    #if !ZERO
		CHECK_CUDA_ERROR
		#endif
		TM_update.start();
		fill_data(d_matrix_host, N);
		TM_update.stop();
		
		#if ZERO
		CHECK_CUDA_ERROR
		#endif
		TM.stop();
    }
	#if ZERO
    CHECK_CUDA_ERROR
	#endif
	TM_app.stop();
	std::cout << "AVG UPDATE:  " << TM_update.total_duration()/int((MINUTES*60*1000)/33.3) << std::endl;
	if (ZERO) 
		TM_app.print("App run time ZC: ");
	else 
		TM_app.print("App run time UM: ");
	std::cout << "AVG APP:  " << TM_app.duration()/int((MINUTES*60*1000)/33.3) << std::endl;
    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    #if ZERO
    SAFE_CALL(cudaFreeHost(d_matrix));
    #else
    SAFE_CALL(cudaFree(d_matrix_host));
    #endif
    
    // -------------------------------------------------------------------------
    cudaDeviceReset();

}

