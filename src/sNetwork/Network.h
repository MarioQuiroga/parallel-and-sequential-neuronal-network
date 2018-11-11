#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "../loaderMnist.h"
#include "../common/utilsCommon.h"
#include "../common/loaderMnist.h"
#include "utils.h"

using namespace std; 

class Network
{
	int numLayers; // Number of layer
	vector<int> sizes; // Number of neurons by layer
	
	// Neuron data
	vector<vector<vector<double>>> weights;  //Vector de pesos
	vector<vector<double>> bias; // Vector sesgos
	vector<vector<double>> inputs; //Input de cada neurona
	vector<vector<double>> outputs; //Output de cada neurona
	vector<vector<double>> deltas; //Error de cada neurona			

	/**	
	*	INICIALIZA LOS PESOS ENTRE 0 Y 1 
	*	LOS PESOS SE GUARDAN EN LA VARIABLE weights
	*
	**/
	void initNetwork()
		{	
			srand(time(NULL)); //Semilla para generar numeros aleatorios
		//Inicializo los pesos 
			//Recordar que el peso W[k][j][i] une la neurona j de la capa k con la neurona i de la capa k+1			
			for(int k=0; k<numLayers-1;k++)
			{
				vector<vector<double>> filas;
				for(int j=0;j<sizes[k];j++)
				{
					vector<double> columnas;					
					for(int i=0;i<sizes[k+1];i++)
					{							
							double aux = (1+(double)(rand() % 10))/1000; 
							columnas.push_back(aux);
					}				
					filas.push_back(columnas);
				}				
				weights.push_back(filas);
			}
		//Inicializo vectores de input, outputs, bias y deltas (errores)
			for (int i=0; i<numLayers; i++)
			{	
				vector<double>	outputs_filas;		
				vector<double>	inputs_filas;		
				vector<double>	deltas_filas;		
				vector<double>	bias_filas;
				for (int j=0; j<sizes[i]; j++)
				{
					//Se llenan con un valor cualquiera					
					double aux = (1+(double)(rand() % 10))/1000; //GENERO ALEATORIOS ENTRE 0 Y 1
					bias_filas.push_back(aux);
					outputs_filas.push_back(j);
					inputs_filas.push_back(j);
					deltas_filas.push_back(j);
				}
				bias.push_back(bias_filas);
				outputs.push_back(outputs_filas);
				inputs.push_back(inputs_filas);
				deltas.push_back(deltas_filas);				
			}
		}
		
		/**	
		*	Calcula la salida de la red para una entrada					
		*
		**/
		vector<double> feedForward(vector <double> & input)		
		{			
			for(int i=0; i<outputs[0].size(); i++) //Calcular el output de la capa de entrada
			{
				outputs[0][i] = input[i];
			}			
			for(int l=1; l<outputs.size(); l++) //Calcular el output de las capas ocultas y salidas
			{
				for(int i=0; i<outputs[l].size(); i++)
				{
					inputs[l][i] = 0;
					for(int j = 0; j< outputs[l-1].size(); j++)
					{
						//Recordar que el peso W[k][j][i] une la neurona 
						//j de la capa k con la neurona i de la capa k+1			
						inputs[l][i] +=  weights[l-1][j][i] * outputs[l-1][j];
					}
					inputs[l][i] += bias[l][i];
					outputs[l][i] = sigmoid(inputs[l][i]); 					
				}
			}
			vector<double> salidas = outputs[outputs.size()-1];
			return salidas;
		}

	public:		
		/**
		*	Constructor de la red
		*	size: vector con la cantidad de neuronas por capa, incluyendo la capa de entrada
		*
		**/		
		Network(vector<int> size)
		{
			sizes = size;
			numLayers = sizes.size();  //Devuelve tamaño del vector;
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
			double suma; 	
			double ERRORANT = 0;
			int contadorEpocas = 0;
			double ERROR = 200.0;	double error;			
			while(ERROR > errorMinimo && contadorEpocas < epocas)
			{					
				ERROR = 0;
				for(int e=0; e<cantidadEjemplos; e++) //	Por cada ejemplo en el conjunto de entrenamiento				
				{
					vector<double> salida = feedForward(x_train[e].input_data);//  Calculo el output para el ejemplo e					
					//Calculo el error en la capa de salida
					for(int i=0; i<outputs[numLayers-1].size(); i++)
					{							
						error = (x_train[e].output[i] - outputs[outputs.size()-1][i]);										
						deltas[deltas.size()-1][i] = sigmoid_prima(inputs[inputs.size()-1][i]) * error;						
						ERROR += error*error; //Guardamos el cuadrado del error
					}
					//Propagar el error hacia atras
					for(int l=numLayers-2; l>=0; l--)
					{
						for (int j=0; j<outputs[l].size(); j++)
						{
							suma = 0;
							for(int i=0; i<weights[l][j].size(); i++)
							{
								suma = suma + (weights[l][j][i]*deltas[l+1][i]);
							}							
							deltas[l][j] = (sigmoid_prima(inputs[l][j]) * suma);
							
							//Actualizar los pesos
							for(int i=0; i<outputs[l+1].size(); i++)
							{								
								weights[l][j][i] = weights[l][j][i] + rateLearning * outputs[l][j] * deltas[l+1][i];
							}
						}						
					} 
					//Actualizar los sesgos					
					for(int l=1; l<numLayers; l++)
					{
						for(int i=0; i<bias[l].size(); i++)
						{
							bias[l][i] = bias[l][i] + rateLearning * deltas[l][i];
						}
					}					
				}										
				ERROR *=  0.5 * (1.0/cantidadEjemplos); // Promedio del error de todos los ejemplos de entrenamiento
				cout.precision(100);
				cout << "Epoca: " << contadorEpocas <<"	ERROR: " << ERROR << endl;
				//CAMBIA EL VALOR DE LA VELOCIDAD DE APRENDIZAJE
				if(ERROR>=ERRORANT)
				{
					rateLearning = rateLearning * 1/(1 + (double)(rand() % 10));
				}
				ERRORANT = ERROR;
				contadorEpocas++;
			}
		}	

		/***
		*	Testear la red
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
				vector<double> salida = feedForward(x_test[i].input_data);
				
				int sal = index_max(salida);
				
				cout << "Salida deseada: " << x_test[i].label << endl;
				for(int j=0;j<x_test[i].output.size();j++)
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
			//return suma/x_test.size();
		}	
	
		void mostrar_pesos()
		{
			cout << endl << "Pesos" << endl;
			for(int k=0; k<weights.size(); k++)
			{
				cout << endl << "[" << k << "]	" << endl;
				for(int j=0; j<weights[k].size(); j++)
				{
					cout << "(";
					for (int i=0; i<weights[k][j].size()-1; i++)
					{
						cout << weights[k][j][i] << ", ";
					}				
					cout << weights[k][j][weights[k][j].size()-1] << ") 	" << endl;					
				}				
			}
			cout << endl;
		}
		
		void mostrar_output()
		{
			cout << endl << "Output" << endl;;
			for(int k=0; k<outputs.size();k++)
			{
				cout << endl << "Capa [" << k << "]:	" << endl;
				for(int j=0; j<outputs[k].size(); j++)
				{					
					cout << "[" << j << "]:(" << inputs[k][j] << ")	" << "[" << j << "]:(" << outputs[k][j] << ") " << endl;					
				}				
			}
			cout << endl;
		}		
		
		void save(string path_to_save)
		{
			ofstream file_to_save (path_to_save, ios::binary);
			if(file_to_save.is_open())
			{
				//Guardo la estructura de la red
				file_to_save << numLayers << endl;
				for(int i=0; i<numLayers; i++)
				{
					file_to_save << sizes[i] << endl;
				}
				//Guardo los pesos
				for(int l=0;l<numLayers-1; l++) //Por cada capa
				{
					for(int i=0; i<sizes[l]; i++) //Por cada neurona
					{
						for(int k=0; k<sizes[l+1]; k++)
						{
							file_to_save << weights[l][i][k] << endl;
						}
					}
				}
				//Guardo los sesgos
				for(int l=0; l<numLayers; l++)
				{
					for(int i=0; i<sizes[l]; i++)
					{
						file_to_save << bias[l][i] << endl;
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
				//Leo la estructura de la red
				file_to_load >> numLayers;
				for (int i=0; i<numLayers; i++)
				{
					file_to_load >> sizes[i];
				}
				//Leo los pesos
				for(int l=0; l<numLayers-1; l++)
				{
					for(int i=0; i<sizes[l]; i++)
					{
						for(int j=0; j<sizes[l+1]; j++)
						{
							file_to_load >> weights[l][i][j];
						}
					}
				}
				//Leo los sesgos
				for(int l=0;l<numLayers; l++)
				{
					for(int i=0; i<sizes[l]; i++)
					{
						file_to_load >> bias[l][i];
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