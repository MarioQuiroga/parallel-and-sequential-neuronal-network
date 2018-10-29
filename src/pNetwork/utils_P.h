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

/**
*	KERNELS UTILIZADOS POR feedForward
*
*/
__global__ void outputInLayerInput(double * ptr, double * ptr_input)
{
	int i = blockIdx.x;
	ptr[i] = ptr_input[i];
}

__global__ void outputNeuron(double * ptr_outputs, 
							 double * ptr_inputs, 
							 double * ptr_weights, 
							 double * ptr_bias,
							 int * ptr_sum, 
							 int * ptr_sizes,
							 int * ptr_sum_sizes,
							 int layer)
{
	int i = blockIdx.x;
	int index1 = ptr_sum_sizes[layer] + i;
	ptr_inputs[index1] = 0;
	for(int j = 0; j< ptr_sizes[layer-1]; j++)	//	Hago la sumatoria
	{
		//Recordar que el peso W[k][j][i] une la neurona 
		//j de la capa k con la neurona i de la capa k+1			
		int indexOutput = (ptr_sum_sizes[layer-1]) + j;
		int indexWeights = (ptr_sum[layer-1]) + // ME POSICIONO EN LA CAPA 
						   ((j * ptr_sizes[layer]) + i);  // ME POSICIONO EN EL PESO

		ptr_inputs[index1] +=  ptr_weights[indexWeights] * ptr_outputs[indexOutput];
	}
	ptr_inputs[index1] += ptr_bias[index1];
	ptr_outputs[index1] = sigmoid(ptr_inputs[index1]); 					
}

/**
*	KERNELS UTILIZADOS POR train_backpropagation
*
*/
__global__ void computeErrorExitLayer(ExampleChar * x_train, double * outputs, 
									  double * deltas,  double * ptr_error, int sumNeuron)
{
	int i = blockIdx.x + sumNeuron;
	ptr_error[i] = (x_train.output[i] - outputs[i]);										
	deltas[i] = sigmoid_prima(inputs[i]) * ptr_error[i];						
	ptr_error[i] = ptr_error[i] * ptr_error[i]; //Store square of error
}

__global__ void updateWeights(double * weights, double * outputs, 
							  double * deltas, int * sumas, 
							  int * sumNeuron, int l, int idelta, int j)
{
	int i = blockIdx.x;
	int indexWeights = sumas[l] + ((j * sizes[l+1]) + i);
	int indexDeltas = sumNeuron[l+1] + i;
	weights[indexWeights] = weights[indexWeights] + rateLearning * outputs[idelta] * deltas[indexDeltas];
}

__global__ void backProp(double * outputs, double * weights, double * sumNeuron
						 double * deltas, double rateLearning, double * sizes,
						 double * inputs, int l, int * sumas)
{
	int j = blockIdx.x;
	//Recordar que el peso W[k][j][i] une la neurona 
	//j de la capa k con la neurona i de la capa k+1			
	suma = 0;	
	for(int i=0; i<sizes[l+1]; i++)
	{
		int indexWeights = sumas[l] + ((j * sizes[l+1]) + i);
		int indexDeltas = sumNeuron[l+1] + i;
		suma = suma + (weights[indexWeights] * deltas[indexDeltas]);
	}					
	int idelta = sumNeuron[l] + j;
	deltas[idelta] = (sigmoid_prima(inputs[idelta]) * suma);	
	updateWeights<<<sizes[l+1], 1>>>(weights, outputs, deltas, rateLearning, sumas, sumNeuron, l, idelta, j); 
}

__global__ void updateB(double * bias, double * deltas, double rateLearning, int sumNeuron)
{
	int i = blockIdx.x;
	int index = sumNeuron + i;
	bias[index] = bias[index] + rateLearning * deltas[index];
}
__global__ void updateBias(double * bias, double * deltas, int * sizes, double rateLearning, int * sumNeuron)
{
	int l = blockIdx.x;
	if (l!=1)
	{
		updateB<<<sizes[l], 1>>>(bias, deltas, rateLearning, sumNeuron[l]);
	}
}