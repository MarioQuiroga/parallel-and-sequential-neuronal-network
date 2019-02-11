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
__global__ 
void memRows(T *** ptr, int * sizes)
{
	int k = blockIdx.x;
	ptr[k] = (T **) malloc(sizeof(T*)*sizes[k]);
	for(int j=0; j<sizes[k]; j++)
	{
		ptr[k][j] = (T *) malloc(sizeof(T) * sizes[k+1]);
	}
}

template <typename T>
__global__
void memColumns(T ** ptr, int * sizes)
{
	ptr[blockIdx.x] = (T *) malloc(sizeof(T) * sizes[blockIdx.x]);
}

template <typename T>
__global__
void memColumnsMatrix(T ** ptr, int size)
{
	int i = blockIdx.x;
	ptr[i] = (T *) malloc(sizeof(T) * size);
}

// 	copy to device
template <typename T>
__global__		
void copyWeights(T *** weights, T * w, int * sumas, int * sizes, int numLayers)
{
	int k = blockIdx.x;
	if(k<numLayers-1)
	{
		for (int j = 0; j < sizes[k]; ++j)
		{
			for (int i = 0; i < sizes[k+1]; ++i)
			{
				//print to test
				//printf("[%i][%i][%i] = [%i]    : %f \n", k, j, i, sumas[k] + ((j*sizes[k+1]) + i), w[sumas[k] + ((j*sizes[k+1]) + i)]);

				weights[k][j][i] = w[sumas[k] + ((j*sizes[k+1]) + i)];
			}
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