#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "kernels/initKernels.h"
#include "kernels/trainKernels.h"
#include "kernels/utilsKernels.h"
#include "../common/utilsCommon.h"
#include "../common/utilsTime.h"
//#include "../common/ExampleChar.h"
#include "../common/loaderMnist.h"

// Libraries CUDA C++
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <cuda.h>

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
	double *** weights; 
	double ** bias; 
	double ** inputs; 
	double ** outputs; 
	double ** deltas; 

	void initNetwork()
	{
		
		int sum = 0;
		// Get Number of Neurons in the network
		for (int i = 0; i < numLayers; ++i)
		{
			sum = sum + sizes_h[i];
		}

		sumas_h.push_back(0);
		sumNeuron_h.push_back(0);
		// Sum Weights until layer
		for (int i = 1; i < numLayers; ++i)
		{		
			sumas_h.push_back(sumas_h[i-1] + (sizes_h[i] * sizes_h[i-1]));
		}		
		// Print sumas_h to test
		//cout << sumas_h[sumas_h.size()] << endl;
		/*for (int i = 0; i < numLayers; ++i)
		{
			cout << sumas_h[i]	<< endl;
		}		
		cout << "-----------------------" << endl;*/
		// Sum Neuron until layer
		for(int i=1; i<sizes_h.size(); i++)
		{
			sumNeuron_h.push_back(sumNeuron_h[i-1] + sizes_h[i-1]);						
		}
		// Print sumNeuron to test
		/*for(int i=0; i<sizes_h.size(); i++)
		{
			cout << sumNeuron_h[i] << endl;
		}*/

		cout << "Alloc memory to device" << endl;		
		gpuErrchk(cudaMalloc((double ****)&weights, sizeof(double **) * (numLayers-1)));
		cudaMalloc((double ***)&bias, sizeof(double * ) * numLayers);
		cudaMalloc((double ***)&inputs, sizeof(double * ) * numLayers);
		cudaMalloc((double ***)&outputs, sizeof(double * ) * numLayers);
		cudaMalloc((double ***)&deltas, sizeof(double * ) * numLayers);
		
		memRows<<<numLayers-1, 1>>>(weights, sizes_d);
		memColumns<<<numLayers, 1>>>(bias, sizes_d);
		memColumns<<<numLayers, 1>>>(inputs, sizes_d);
		memColumns<<<numLayers, 1>>>(outputs, sizes_d);
		memColumns<<<numLayers, 1>>>(deltas, sizes_d);

		// Sum weights and bias per layer
		cudaMalloc((int **) &sumas_d, sizeof(int) * sumas_h.size());
		cudaMalloc((int **) &sumNeuron_d, sizeof(int) * sumNeuron_h.size());
		cudaMemcpy(sumas_d, getPointer(sumas_h), sizeof(int) * sumas_h.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(sumNeuron_d, getPointer(sumNeuron_h), sizeof(int) * sumas_h.size(), cudaMemcpyHostToDevice);
		
		// Init sequential weights and bias
		thrust::device_vector<double> w(sumas_h[sumas_h.size()-1]);
		thrust::device_vector<double> b(sum);
		srand(time(NULL)); // Seed for rand
		for(int i = 0; i<sumas_h[sumas_h.size()-1]; i++)
		{
			w[i] = ((1+(double)(rand() % 10))/1000); 
		}
		for(int i = 0; i<sum; i++)
		{
			b[i] = ((1+(double)(rand() % 10))/1000); 
		}
		double * p_w = thrust::raw_pointer_cast(&w[0]);
		double * p_b = thrust::raw_pointer_cast(&b[0]);
		// Kernels copy to device 
		copyWeights<<<numLayers-1,1>>>(weights, p_w, sumas_d, sizes_d, numLayers);		
		matrixCpy<<<numLayers,1>>>(bias, p_b, sumNeuron_d, sizes_d, numLayers);
		if(cudaGetLastError()!=0)
		{
			printf("Error in initNetwork: %s \n", cudaGetErrorName(cudaGetLastError()));	
		}
	}

	/**	
	*	Compute network output for an input
	*	
	**/
	vector<double> feedForward(ExampleChar * input, int i)		
	{			
		// Compute output from input layers		
		outputInLayerInput<<<sizes_h[0], 1>>>(outputs, input, i);
		kernelCheck();
		// Compute output from hidden and output layer
		for(int l=1; l<sizes_h.size(); l++)	//	Go through layers
		{
			outputNeuron<<<sizes_h[l], 1>>>(outputs, inputs, weights, bias, sizes_d, l);
			kernelCheck();
		}
		// Return output 
		vector<double> res(sizes_h[sizes_h.size()-1]);		
		double * out_h = (double*)malloc(sizeof(double)* res.size());
		double * out_d;
		gpuErrchk(cudaMalloc((void **) &out_d, sizeof(double)*res.size()));		
		copyVector<<<1,1>>>(out_d, outputs, numLayers-1, res.size());
		kernelCheck();
		gpuErrchk(cudaMemcpy(out_h, out_d, sizeof(double) * res.size(), cudaMemcpyDeviceToHost));		
		for (int j = 0; j < res.size(); ++j)
		{
			res[j] = out_h[j];
			//cout << res[j] << "---";
		}
		cout << endl;
		return res;
	}

	void feedForwardTrain(ExampleChar * input, int i)		
	{			
		// Compute output from input layer		
		outputInLayerInput<<<sizes_h[0], 1>>>(outputs, input, i);		
		kernelCheck();

		// Compute output from hidden and output layer
		for(int l=1; l<sizes_h.size(); l++)	//	Go through layers
		{
			outputNeuron<<<sizes_h[l], 1>>>(outputs, inputs, weights, bias, sizes_d, l);
			kernelCheck();
		}
	}
	
	public:
		Network_P(vector<int> size)
		{
			numLayers = size.size(); 
			sizes_h = size;  // To sizes in host
			int nBytes = sizeof(int) * numLayers;
			cudaMalloc((int **)&sizes_d, nBytes);
			int * ptr_size = getPointer(size);
			cudaMemcpy(sizes_d, ptr_size, nBytes, cudaMemcpyHostToDevice); // To sizes in device			
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
		vector<double> train_backpropagation(vector<ExampleChar> x_train, double rateLearning, int epocas, double errorMinimo, int cantidadEjemplos)
		{	
			std::vector<double> response;
			time_t first, second;
			cout << "Training Network..." << endl;
			double ERRORANT = 0;
			int contadorEpocas = 0;
			double ERROR = 200.0;		
			// Copy train set to memory device
			ExampleChar * d_x_train;
			gpuErrchk(cudaMalloc((void**) & d_x_train, cantidadEjemplos * sizeof (ExampleChar)));						
			ExampleChar * x = (ExampleChar*) malloc(sizeof(ExampleChar)*cantidadEjemplos);
			copyExamples(x, x_train, cantidadEjemplos);
			gpuErrchk(cudaMemcpy(d_x_train, x, cantidadEjemplos * sizeof (ExampleChar), cudaMemcpyHostToDevice));			
			//printData<<<1,1>>>(d_x_train);			
			free(x);
			double * error;
			gpuErrchk(cudaMalloc((void**) &error, sizes_h[sizes_h.size()-1] * sizeof (double)));						
			double * er = (double*) malloc(sizeof(double)*sizes_h[sizes_h.size()-1]);
			while(ERROR > errorMinimo && contadorEpocas < epocas)
			{			
				first = time(NULL);  
				ERROR = 0;
				for(int e=0; e<cantidadEjemplos; e++) // For each example from d_x_train				
				{					
					feedForwardTrain(d_x_train, e);						
					computeErrorExitLayer<<<sizes_h[numLayers-1], 1>>>(d_x_train, 
											   outputs,
											   inputs, 
											   deltas, 
											   error, 
											   numLayers-1,
											   e);
					kernelCheck();
					updateBias<<<sizes_h[numLayers-1], 1>>>(bias, deltas, rateLearning, numLayers-1);
					kernelCheck();									
					gpuErrchk(cudaMemcpy(er, error, sizeof(double)*sizes_h[sizes_h.size()-1], cudaMemcpyDeviceToHost));					
					//cout << "Errores";
					for(int i=0;i<sizes_h[sizes_h.size()-1];i++)
					{						
						ERROR+= er[i];
						//cout << er[i] << "		";
					}
					//cout << endl;
					//Backpropagation of Error							
					for(int l=numLayers-2; l>=0; l--)
					{
						backPropagationError<<<sizes_h[l], 1>>>(weights, deltas, sizes_d, inputs, l);
						kernelCheck();			
						// Update Weights	
						//cout << l << ": " << sizes_h[l] << endl;					
						//cout << l << ": " << sizes_h[l+1] << endl;					
						updateWeights<<<sizes_h[l], sizes_h[l+1]>>>(weights, outputs,deltas, rateLearning, l); 
						kernelCheck();									
						//Update Bias
						updateBias<<<sizes_h[l], 1>>>(bias, deltas, rateLearning, l);
						kernelCheck();									
					}							
				}														
				ERROR *=  0.5 * (1.0/cantidadEjemplos); // Average error of examples
				cout.precision(100);
				cout << "Epoca: " << contadorEpocas <<"	ERROR: " << ERROR << endl;
				response.push_back(ERROR);
				//Change rateLearning
				if(ERROR>=ERRORANT)
				{
					rateLearning = rateLearning * 1/(1 + (double)(rand() % 10));
				}
				ERRORANT = ERROR;
				contadorEpocas++;
				second = time(NULL);
				tm train_time = getTm(difftime(second, first));
				cout << "Tiempo epoca: " << train_time.tm_hour << ":" << train_time.tm_min << ":" << train_time.tm_sec << endl;

			}
			gpuErrchk(cudaFree(d_x_train));			
			gpuErrchk(cudaFree(error));
			free(er);
			return response;
		}	

		/***
		*	Testing Network
		*	-----------------------------------------------------------------
		* 	Nota: En la salida de la red, la neurona con mayor valor será la que
		*	esté activa, y la que determinará el resultado según su posición.
		*	
		*	x_test: struct ExampleChar, "Ver archivo loaderMnist.h"
		*	cantidadEjemplos: cantidad de ejemplos a testear para no tener que cambiar el vector de prueba	
		*
		***/
		double test_network(vector<ExampleChar> x_test, int cantidadEjemplos)
		{	
			ExampleChar * d_x_test;	
			gpuErrchk(cudaMalloc((void**) & d_x_test, cantidadEjemplos * sizeof (ExampleChar)));
			ExampleChar * x = (ExampleChar*) malloc(sizeof(ExampleChar)*cantidadEjemplos);
			copyExamples(x, x_test, cantidadEjemplos);
			gpuErrchk(cudaMemcpy(d_x_test, x, cantidadEjemplos * sizeof (ExampleChar), cudaMemcpyHostToDevice));
			free(x);			
			double suma = 0;
			for(int i=0; i<cantidadEjemplos;i++)
			{
				vector<double> salida = feedForward(d_x_test, i);				
				int sal = index_max(salida);				
				//cout << "Salida deseada: " << x_test[i].label << endl;
				/*for(int j=0;j<sizes_h[sizes_h.size()-1];j++)
				{					
					cout << x_test[i].output[j] << "|";
				}
				cout << endl;*/
				
				/*cout << "Salida obtenida: " << sal << endl;
				cout.precision(100);
				for(int j=0;j<salida.size();j++)
				{				
					cout << salida[j] << endl;
				}				
				cout << endl;*/
				
				if (sal == x_test[i].label)
				{
					suma++;
				}
				//cout << "------------------------------------------------" << endl;
			}
			cout << "Presicion: " <<  suma/cantidadEjemplos*100 << "%" <<  endl;
			return suma/cantidadEjemplos;
		}	
	
		void mostrar_pesos()
		{
			double * w_d;
			gpuErrchk(cudaMalloc((void **)&w_d, sizeof(double)));
			double w_h;
			cout << endl << "Pesos" << endl;
			for(int k=0; k<numLayers-1; k++)
			{
				cout << endl << "[" << k << "]	" << endl;
				for(int j=0; j<sizes_h[k]; j++)
				{
					cout << "(";
					for (int i=0; i<sizes_h[k+1]; i++)
					{
						
						copyWeight<<<1, 1>>>(w_d, weights, k, j, i, 0);
						kernelCheck();
						gpuErrchk(cudaMemcpy((void*)&w_h, (void*) w_d, sizeof(double), cudaMemcpyDeviceToHost));		
						//printf("Error: %s \n", cudaGetErrorName(cudaGetLastError()));				
						cout << w_h << ", ";						
					}				
					cout << endl;					
				}				
			}
			cout << endl;
			cudaFree(w_d);
		}
		
		void save(string path_to_save)
		{
			ofstream file_to_save (path_to_save, ios::binary);
			if(file_to_save.is_open())
			{
				double * w_d;
				gpuErrchk(cudaMalloc((double **)&w_d, sizeof(double)));
				double w_h;
				// Store network structure
				file_to_save << numLayers << endl;
				for(int i=0; i<numLayers; i++)
				{
					file_to_save << sizes_h[i] << endl;
				}
				// Store Weights
				for(int l=0;l<numLayers-1; l++) // For each layer
				{
					for(int i=0; i<sizes_h[l]; i++) // For each neuron
					{
						for(int k=0; k<sizes_h[l+1]; k++)
						{							
							copyWeight<<<1, 1>>>(w_d, weights, l, i, k, 0);							
							kernelCheck();
							gpuErrchk(cudaMemcpy(&w_h, w_d, sizeof(double), cudaMemcpyDeviceToHost));
							file_to_save << w_h << endl;							
						}
					}
				}
				gpuErrchk(cudaFree(w_d));
				// Store Bias
				double * b_d;
				gpuErrchk(cudaMalloc((double **)&b_d, sizeof(double)));
				double b_h;
				for(int l=0; l<numLayers; l++)
				{
					for(int i=0; i<sizes_h[l]; i++)
					{						
						copyBias<<<1,1>>>(b_d, bias, l, i, 0);				
						kernelCheck();
						gpuErrchk(cudaMemcpy(&b_h, b_d, sizeof(double), cudaMemcpyDeviceToHost));
						file_to_save << b_h << endl;
					}
				}			
				gpuErrchk(cudaFree(b_d));
				file_to_save.close();
				cout << "Network saved" << endl;		
			}
			else
			{
				cout << "Error to create file" << endl;
			}			
		}
		
		void load(string path_to_load)
		{
			ifstream file_to_load (path_to_load, ios::binary);			
			if(file_to_load.is_open())
			{
				double * w_d;
				gpuErrchk(cudaMalloc((double **)&w_d, sizeof(double)));
				double w_h;
				// Read network structure
				file_to_load >> numLayers;
				for (int i=0; i<numLayers; i++)
				{
					file_to_load >> sizes_h[i];
				}
				// Read Weights
				for(int l=0; l<numLayers-1; l++)
				{
					for(int j=0; j<sizes_h[l]; j++)
					{
						for(int i=0; i<sizes_h[l+1]; i++)
						{								
							file_to_load >> w_h;							
							gpuErrchk(cudaMemcpy(w_d, &w_h, sizeof(double), cudaMemcpyHostToDevice));
							copyWeight<<<1, 1>>>(w_d, weights, l, j, i, 1);							
							kernelCheck();
						}
					}
				}
				gpuErrchk(cudaFree(w_d));
				// Read Bias
				double * b_d;
				gpuErrchk(cudaMalloc((double **)&b_d, sizeof(double)));
				double b_h;
				for(int l=0;l<numLayers; l++)
				{
					for(int i=0; i<sizes_h[l]; i++)
					{						
						file_to_load >> b_h;						
						gpuErrchk(cudaMemcpy(b_d, &b_h, sizeof(double), cudaMemcpyHostToDevice));
						copyBias<<<1, 1>>>(b_d, bias, l, i, 1);									
						kernelCheck();
					}
				}
				gpuErrchk(cudaFree(b_d));
				file_to_load.close();
				cout << "Network loaded" << endl;				
			}
			else
			{
				cout << "Error to open file" << endl;
			}
			
		}
};
