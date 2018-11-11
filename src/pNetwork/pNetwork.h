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
		srand(time(NULL)); // Seed for rand
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
		cudaMalloc((double ****)&weights, sizeof(double **) * (numLayers-1));
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
	}

	/**	
	*	Compute network output for an input
	*	
	**/
	vector<double> feedForward(ExampleChar * input)		
	{			
		// Compute output from input layers
		outputInLayerInput<<<sizes_h[0], 1>>>(outputs, input[0].input_data);

		// Compute output from hidden and output layer
		for(int l=1; l<sizes_h.size(); l++)	//	Go through layers
		{
			outputNeuron<<<sizes_h[l], 1>>>(outputs, inputs, weights, bias, sizes_d, l);
		}
		// Return output 
		vector<double> res(sizes_h[sizes_h.size()-1]);
		cudaMemcpy(getPointer(res), outputs[numLayers-1], sizeof(double) * res.size(), cudaMemcpyDeviceToHost);
		return res;
	}

	void feedForwardTrain(ExampleChar * input)		
	{			
		// Compute output from input layer		
		outputInLayerInput<<<sizes_h[0], 1>>>(outputs, input[0].input_data);		

		// Compute output from hidden and output layer
		for(int l=1; l<sizes_h.size(); l++)	//	Go through layers
		{
			outputNeuron<<<sizes_h[l], 1>>>(outputs, inputs, weights, bias, sizes_d, l);
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
		void train_backpropagation(vector<ExampleChar> x_train, double rateLearning, int epocas, double errorMinimo, int cantidadEjemplos)
		{	
			cout << "Entrenando..." << endl;
			double ERRORANT = 0;
			int contadorEpocas = 0;
			double ERROR = 200.0;		
			// Copy train set to memory device
			ExampleChar * d_x_train;
			cudaMalloc((void**) & d_x_train, cantidadEjemplos * sizeof (ExampleChar));
			cudaMemcpy(&d_x_train, getPointer(x_train), cantidadEjemplos * sizeof (ExampleChar), cudaMemcpyHostToDevice);
			while(ERROR > errorMinimo && contadorEpocas < epocas)
			{					
				ERROR = 0;
				for(int e=0; e<cantidadEjemplos; e++) // For each example from d_x_train				
				{					
					feedForwardTrain(d_x_train);	
					thrust::device_vector<double> error(sizes_h[sizes_h.size()-1]);			
					double * ptr_error = (double *)thrust::raw_pointer_cast(&error);
					computeErrorExitLayer<<<sizes_h[numLayers-1], 1>>>(d_x_train, 
																	   outputs,
																	   inputs, 
																	   deltas, 
																	   ptr_error, 
																	   numLayers-1);
					ERROR += thrust::reduce(error.begin(), error.end(), 0, thrust::plus<double>());					

					//Backpropagation of Error							
					for(int l=numLayers-2; l>=0; l--)
					{
						backPropagationError<<<sizes_h[l], 1>>>(weights, deltas, 
																sizes_d, inputs, l);
						// Update Weights						
						updateWeights<<<sizes_h[l], sizes_h[l+1]>>>(weights, outputs, 
														   			deltas, rateLearning, 
														   			l); 
						//Update Bias
						updateBias<<<sizes_h[l], 1>>>(bias, deltas, rateLearning, l);
					}							
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
			//cudaFree(d_x_train);
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
		void test_network(vector<ExampleChar> x_test, int cantidadEjemplos)
		{			
			double suma = 0;
			for(int i=0; i<cantidadEjemplos;i++)
			{
				vector<double> salida = feedForward(&x_test[i]);
				
				int sal = index_max(salida);
				
				cout << "Salida deseada: " << x_test[i].label << endl;
				for(int j=0;j<sizes_h[sizes_h.size()-1];j++)
				{					
					cout << x_test[i].output[j] << "|";
				}
				cout << endl;
				
				cout << "Salida obtenida: " << sal << endl;
				cout.precision(100);
				for(int j=0;j<salida.size();j++)
				{				
					cout << salida[j] << endl;
				}				
				cout << endl;
				
				if (sal == x_test[i].label)
				{
					suma++;
				}
				cout << "------------------------------------------------" << endl;
			}
			cout << "Presicion: " <<  suma/cantidadEjemplos*100 << "%" <<  endl;
		}	
	
		void mostrar_pesos()
		{
			cout << endl << "Pesos" << endl;
			for(int k=0; k<numLayers-1; k++)
			{
				cout << endl << "[" << k << "]	" << endl;
				for(int j=0; j<sizes_h[k]; j++)
				{
					cout << "(";
					for (int i=0; i<sizes_h[k+1]; i++)
					{
						double * w_d;
						cudaMalloc((double **)&w_d, sizeof(double));
						copyWeight<<<1, 1>>>(w_d, weights, k, j, i, 0);
						double w_h;
						cudaMemcpy(&w_d, &w_h, sizeof(double), cudaMemcpyDeviceToHost);						
						cout << w_h << ", ";
						cudaFree(w_d);
					}				
					cout << endl;					
				}				
			}
			cout << endl;
		}
		
		void save(string path_to_save)
		{
			ofstream file_to_save (path_to_save, ios::binary);
			if(file_to_save.is_open())
			{
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
							double * w_d;
							cudaMalloc((double **)&w_d, sizeof(double));
							copyWeight<<<1, 1>>>(w_d, weights, l, i, k, 0);
							double w_h;
							cudaMemcpy(&w_d, &w_h, sizeof(double), cudaMemcpyDeviceToHost);
							file_to_save << w_h << endl;
							cudaFree(w_d);
						}
					}
				}
				// Store Bias
				for(int l=0; l<numLayers; l++)
				{
					for(int i=0; i<sizes_h[l]; i++)
					{
						double * b_d;
						cudaMalloc((double **)&b_d, sizeof(double));
						copyBias<<<1,1>>>(b_d, bias, l, i, 0);
						double b_h;
						cudaMemcpy(&b_d, &b_h, sizeof(double), cudaMemcpyDeviceToHost);
						file_to_save << b_h << endl;
						cudaFree(b_d);
					}
				}			
				file_to_save.close();
				cout << "La red se ha guardado correctamente" << endl;		
			}
			else
			{
				cout << "Error al crear el archivo" << endl;
			}			
		}
		
		void load(string path_to_load)
		{
			ifstream file_to_load (path_to_load, ios::binary);			
			if(file_to_load.is_open())
			{
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
							double w_h;
							file_to_load >> w_h;
							double * w_d;
							cudaMalloc((double **)&w_d, sizeof(double));
							cudaMemcpy(&w_d, &w_h, sizeof(double), cudaMemcpyHostToDevice);
							copyWeight<<<1, 1>>>(w_d, weights, l, j, i, 1);							
							cudaFree(w_d);
						}
					}
				}
				// Read Bias
				for(int l=0;l<numLayers; l++)
				{
					for(int i=0; i<sizes_h[l]; i++)
					{
						double b_h;
						file_to_load >> b_h;
						double * b_d;
						cudaMalloc((double **)&b_d, sizeof(double));
						cudaMemcpy(&b_d, &b_h, sizeof(double), cudaMemcpyHostToDevice);
						copyBias<<<1, 1>>>(b_d, bias, l, i, 1);							
						cudaFree(b_d);
					}
				}
				file_to_load.close();
				cout << "La red se ha cargado correctamente" << endl;				
			}
			else
			{
				cout << "Error al abrir el archivo. Verifique la ruta" << endl;
			}
			
		}
};
