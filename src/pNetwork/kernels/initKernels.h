#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std; 

/**
*	INIT KERNELS 
*
*/
// 	alloc memory in device
template <typename T>
__device__
void memColumnsDevice(T ** ptr, int size)
{
	ptr[blockIdx.x] = (T *) malloc(sizeof(T) * size);
}

template <typename T>
__global__ 
void memRows(T *** ptr, int * sizes)
{
	int i = blockIdx.x;
	ptr[i] = (T **) malloc(sizeof(T*)*sizes[i]);
	memColumnsDevice(ptr[i], sizes[i+1]);
}

template <typename T>
__global__
void memColumns(T ** ptr, int * sizes)
{
	ptr[blockIdx.x] = (T *) malloc(sizeof(T) * sizes[blockIdx.x]);
}

// 	copy to device
template <typename T>
__global__		
void copyWeights(T *** weights, T * w, int * sumas, int * sizes)
{
	int k = blockIdx.x;
	for (int j = 0; j < sizes[k]; ++j)
	{
		for (int i = 0; i < sizes[k+1]; ++i)
		{
			weights[k][j][i] = w[sumas[k] + (sizes[k]*sizes[k+1] + i)];
		}
	}
}

template <typename T>
__global__ 
void matrixCpy(T ** bias, T * b, int * sumNeuron, int * sizes, int numLayers)
{
	int i = blockIdx.x;
	if(i<numLayers)
	{
		for (int j = 0; j < sizes[i]; ++j)
		{
			// print to test
			//printf(" [%i][%i]:[%i] = %f \n",i,j,sumNeuron[i]+j, b[sumNeuron[i]+j]);
			bias[i][j] = b[sumNeuron[i]+j];
		}
	}
}