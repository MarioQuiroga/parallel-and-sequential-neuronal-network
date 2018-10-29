#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "../loaderMnist.h"
#include "utils_P.h"

// Libraries CUDA C++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

using namespace std; 

class Network_P
{
	// Host Data
	int numLayers; // Number of layer
	vector<int> sizes_h;
	vector<int> sumas_h;
	vector<int> sumNeuron_h;
	// Device Data
	int * sizes_d;// Number of neurons by layer	
	int * sumas_d; // Sum of number of elements per layer
	int * sumNeuron_d; 

	// Neuron data
	double * weights; // Vector de pesos
	double * bias; // Vector sesgos
	double * inputs; // Input de cada neurona
	double * outputs; // Output de cada neurona
	double * deltas; // Error de cada neurona			

	void initNetwork()
	{
		srand(time(NULL)); // Seed for rand
		int sum = 0;
		for (int i = 0; i < numLayers; ++i)
		{
			sum = sum + sizes_h[i];

		}		
		sumas_h[0] = sizes_h[0] * sizes_h[1];
		sumNeuron_h[0] = sizes_h[0];
		for (int i = 1; i < numLayers-1; ++i)
		{			
			sumas_h[i] = sumas_h[i-1] + (sizes_h[i] * sizes_h[i+1]);			
			sumNeuron_h[i] = sumNeuron_h[i-1] + sizes_h[i];			
		}		
		cudaMalloc((int *) &sumas_d, sizeof(int) * sumas_h.size());
		cudaMalloc((int *) &sumNeuron_d, sizeof(int) * sumNeuron_h.size());
		cudaMemcpy(sumas_d, getPointer(sumNeuron_h), sizeof(int) * sumas_h.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(sumNeuron_d, getPointer(sumNeuron_h), sizeof(int) * sumas_h.size(), cudaMemcpyHostToDevice);

		cudaMalloc((double *)&weights, sizeof(double) * sumas_h[sumas_h.size()-1]);
		cudaMalloc((double *)&bias, sizeof(double) * sum);
		cudaMalloc((double *)&inputs, sizeof(double) * sum);
		cudaMalloc((double *)&outputs, sizeof(double) * sum);
		cudaMalloc((double *)&deltas, sizeof(double) * sum);
		
		// Init sequential weights and bias
		std::vector<double> w(sumas_h[sumas_h.size()-1]);
		std::vector<double> b(sum);
		for(int i = 0; i<sumas_h[sumas_h.size()-1]; i++)
		{
			w.push_back((1+(double)(rand() % 10))/1000); 
		}
		for(int i = 0; i<sum; i++)
		{
			b.push_back((1+(double)(rand() % 10))/1000); 
		}
		cudaMemcpy(weights, getPointer(w), w.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(bias, getPointer(b), b.size(), cudaMemcpyHostToDevice);
		// Probar si es necesario !!
		cudaMemcpy(inputs, getPointer(b), b.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(outputs, getPointer(b), b.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(deltas, getPointer(b), b.size(), cudaMemcpyHostToDevice);

	}

	/**	
	*	Compute network output for an input
	*	
	**/
	vector<double> feedForward(ExampleChar * input)		
	{			
		// Compute output from input layers
		outputInLayerInput<<<sizes_h[0], 1>>>(outputs, input.input_data);

		// Compute output from hidden and output layer
		for(int l=1; l<sizes_h.size(); l++)	//	Go through layers
		{
			outputNeuron<<<sizes[l], 1>>>(outputs, inputs, weights, bias, 
										  sumas_d, sizes_d, sumNeuron_d, l);
		}
		vector<double> res(sizes[sizes.size()-1]);
		int index = sumNeuron_h[sumNeuron_h.size()-2] + 1;
		cudaMemcpy(getPointer(res), outputs[index], sizeof(double) * res.size(), cudaMemcpyDeviceToHost);
		return res;
	}

	void feedForwardTrain(ExampleChar * input)		
	{			
		// Compute output from input layer		
		outputInLayerInput<<<sizes_h[0], 1>>>(outputs, input.input_data);		

		// Compute output from hidden and output layer
		for(int l=1; l<sizes_h.size(); l++)	//	Go through layers
		{
			outputNeuron<<<sizes_h[l], 1>>>(outputs, inputs, weights, bias, 
										  sumas_d, sizes_d, sumNeuron_d, l);
		}
	}
	
	public:
		Network_P(vector<int> size)
		{
			numLayers = size.size(); 
			sizes_h = size;  // To sizes in host
			int nBytes = sizeof(int) * numLayers;
			cudaMalloc((int *)&sizes, nBytes);
			int * ptr_size = getPointer(size);
			cudaMemcpy(sizes, ptr_size, nBytes, cudaMemcpyHostToDevice); // To sizes in device
			initNetwork();			
		}

		/**
		*
		*	Aprendizaje por Retropropagación
		*	--------------------------------
		*	x_train: vector de ejemplos de entrenamiento (struct ExampleChar, "Ver archivo loaderMnist.h")
		*	rateLearning: Velocidad de aprendizaje
		*	epocas: Cantidad de iteraciones maxima que se prensenta el conjunto de entrenamiento a la red
		*	errorMinimo: El entrenamiento finaliza cuando se ERROR =< errorMinimo
		*
		**/
		void train_backpropagation(ExampleChar * x_train, double rateLearning, int epocas, double errorMinimo, int cantidadEjemplos)
		{	
			cout << "Entrenando..." << endl;
			double suma; 	
			double ERRORANT = 0;
			int contadorEpocas = 0;
			double ERROR = 200.0;	double error;		
			// Copy train set to memory device
			ExampleChar * d_x_train;
			cudaMalloc((void**) & d_x_train, cantidadEjemplos * sizeof (ExampleChar));
			cudaMemcpy(d_x_train, x_train, cantidadEjemplos * sizeof (ExampleChar), cudaMemcpyHostToDevice);
			while(ERROR > errorMinimo && contadorEpocas < epocas)
			{					
				ERROR = 0;
				for(int e=0; e<cantidadEjemplos; e++) // For each example from d_x_train				
				{					
					feedForward(d_x_train[e]);	
					thrust::device_vector<double> error(sizes[sizes.size-1]);			
					double * ptr_error = thrust::raw_pointer_cast(&error);
					computeErrorExitLayer<<<sizes[numLayers-1], 1>>>(d_x_train[e], 
																	 outputs, 
																	 deltas, 
																	 ptr_error, 
																	 sumNeuron_h[sumNeuron_h.size()-2]);
					ERROR += thrust::reduce(error.begin(), error.end(), 0, thrust::plus<double>());					

					//Backpropagation of Error							
					for(int l=numLayers-2; l>=0; l--)
					{
						backProp<<<sizes_h[l], 1>>>(outputs, weights, sumNeuron_d 
										  deltas, rateLearning, sizes_d,
										  inputs, l, sumas_d);
					} 
					//Update Bias
					updateBias<<<numLayers, 1>>>(bias, deltas, sizes_d, rateLearning, sumNeuron_d);			
				}														
				ERROR *=  0.5 * (1.0/cantidadEjemplos); // Average error of examples
				cout.precision(100);
				cout << "Epoca: " << contadorEpocas <<"	ERROR: " << ERROR << endl;
				//Change rateLearning
				if(ERROR>=ERRORANT)
				{
					rateLearning = rateLearning * 1/(1 + (double)(rand() % 10));
				}
				ERRORANT = ERROR;
				contadorEpocas++;
			}
		}	

		void mostrar_pesos()
		{
			cout << endl << "Pesos" << endl;
			for(int k=0; k<numLayers-1; k++)
			{
				cout << endl << "[" << k << "]	" << endl;
				for(int j=0; j<sizes_h[k+1]; j++)
				{
					cout << "(";
					for (int i=0; i<sizes[k]; i++)
					{
						int index = sumas_h[k] + ((j * sizes[k+1]) + i);
						cout << weights[index] << ", ";
					}				
					cout <<  ") 	" << endl;					
				}				
			}
			cout << endl;
		}
		
		void mostrar_output()
		{
			cout << endl << "Output" << endl;;
			for(int k=0; k<numLayers;k++)
			{
				cout << endl << "Capa [" << k << "]:	" << endl;
				for(int j=0; j<sizes_h[k]; j++)
				{		
					int index = sumNeuron_h[k] + j;
					cout << "[" << j << "]:(" << inputs[index] << ")	" << "[" << j << "]:(" << outputs[index] << ") " << endl;					
				}				
			}
			cout << endl;
		}		
};