#ifndef utilsKernels_h
#define utilsKernels_h
#include <stdio.h>
#include <cmath>
#include <vector>

using namespace std; 

template <typename T>
T * getPointer(std::vector<T> & v)
{
	auto p = v.begin();
	T * ptr = v.data() + (p - v.begin());
	return ptr;
}

__device__ double sigmoid(double x)
{
	return 1/(1+exp(-x));
}
__device__ double sigmoid_prima(double x)
{
	return (sigmoid(x) * (1 - sigmoid(x)));
}

template <typename T>
__global__ void copyWeight(T * w, T *** weights, int k, int j, int i, int flag)
{
	if (flag == 0)
	{
		
		*w = weights[k][j][i];
		
	}
	else
	{
		if (flag == 1)
		{			
			weights[k][j][i] = *w;
		}		
	}		
}

template <typename T>
__global__ void copyBias(T * b, T ** bias, int i, int j, int flag)
{
	if (flag == 0)
	{
		*b = bias[i][j];
	}
	else
	{
		if (flag == 1)
		{
			bias[i][j] = *b;
		}		
	}		
}


#endif