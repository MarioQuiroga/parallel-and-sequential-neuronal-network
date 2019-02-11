#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utilsKernels.h"
#include "../../common/utilsCommon.h"
#include "../../common/ExampleChar.h"

using namespace std; 
/**
*	KERNELS for feedForward
*
*/
__global__ 
void outputInLayerInput(double ** ptr, double ** ptr_input, int index)
{
	int i = blockIdx.x;
	ptr[0][i] = ptr_input[index][i];
}

__global__ 
void outputNeuron(double ** ptr_outputs, 
		  double ** ptr_inputs, 
		  double *** ptr_weights, 
		  double ** ptr_bias,
		  int * ptr_sizes,
		  int layer)
{
	int i = blockIdx.x;
	ptr_inputs[layer][i] = 0;
	for(int j = 0; j< ptr_sizes[layer-1]; j++)	//	Hago la sumatoria
	{
		//Recordar que el peso W[k][j][i] une la neurona 
		//j de la capa k con la neurona i de la capa k+1					
		ptr_inputs[layer][i] +=  ptr_weights[layer-1][j][i] * ptr_outputs[layer-1][j];
	}
	ptr_inputs[layer][i] += ptr_bias[layer][i];
	ptr_outputs[layer][i] = sigmoid(ptr_inputs[layer][i]); 					
}

/**
*	KERNELS for train_backpropagation
*
*/
__global__ void computeErrorExitLayer(double ** y_train, 
				    double ** outputs, 
				    double ** inputs,
    				double ** deltas,  
				    double * ptr_error, 
				    int j,
				    int e)
{
	int i = blockIdx.x;	
	//ptr_error[i] = 1;										
	ptr_error[i] = (y_train[e][i] - outputs[j][i]);										
	deltas[j][i] = sigmoid_prima(inputs[j][i]) * ptr_error[i];						
	ptr_error[i] = ptr_error[i] * ptr_error[i]; //Store square of error
}

__global__ void backPropagationError(double *** weights, double ** deltas, 
									 int * sizes, double ** inputs, int l)
{
	int j = blockIdx.x;
	//Recordar que el peso W[k][j][i] une la neurona 
	//j de la capa k con la neurona i de la capa k+1			
	double suma = 0;	
	for(int i=0; i<sizes[l+1]; i++)
	{		
		suma = suma + (weights[l][j][i]*deltas[l+1][i]);
	}						
	deltas[l][j] = (sigmoid_prima(inputs[l][j]) * suma);		
}

__global__ void updateWeights(double *** weights, double ** outputs, 
							  double ** deltas, double rateLearning,
							  int l)
{
	//int i = blockIdx.x;	
	//int j = threadIdx.x;
	int j = blockIdx.x;	
	int i = threadIdx.x;
	weights[l][j][i] += (rateLearning * outputs[l][j] * deltas[l+1][i]); 
}

__global__ void updateBias(double ** bias, 
						   double ** deltas, 
						   double rateLearning, 
						   int l)
{
	int i = blockIdx.x;
	bias[l][i] += (rateLearning * deltas[l][i]);	
}

__global__ void copy(double * w_h, double *** w, int l, int j, int i)
{
	*w_h = w[l][j][i];
}

void copyExamples(double ** x, double ** y, std::vector<ExampleChar> x_train, int cantidad, int inSize, int outSize)
{
	for (int i = 0; i < cantidad; i++)
	{
		memcpy(x[i], x_train[i].input_data, sizeof(double)*inSize);
		memcpy(y[i], x_train[i].output, sizeof(double)*outSize);
	}
}

 __global__ void printMatrix(double ** in, int filas, int columnas)
 {
 	for (int i = 0; i < filas; i++)
 	{
 		for (int j= 0; j < columnas; j++)
 		{
 			printf("%f|", in[i][j]);
 		}
 		printf("\n");
 	}
 }

 __global__ void memCpyExampleCharTODevice(double ** dst, double * src, int i)
 {
 	dst[i] = src;
 }