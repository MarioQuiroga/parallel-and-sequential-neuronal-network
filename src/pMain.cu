#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "common/loaderMnist.h"
#include "pNetwork/pNetwork.h"
#include "common/utilsTime.h"

int main()
{
	//Constructor MnistLoader
	MnistLoader mnist = MnistLoader("../MNIST/data/train-images.idx3-ubyte", 
									"../MNIST/data/tdata/10k-images.idx3-ubyte", 
									"../MNIST/data/train-labels.idx1-ubyte", 
									"../MNIST/data/t10k-labels.idx1-ubyte");

	int typeNet = 0; // 0 PARALLEL, 1 SEQUENTIAL
	vector<int> sizes;	
	sizes.push_back(784); sizes.push_back(100); sizes.push_back(10);			
	if(typeNet == 0)
	{
		Network_P net = Network_P(sizes);		
	}
	else
	{
		Network net = Network(sizes);		
	}	
	int EPOCAS = 500;
	double ERROR = 0.003;
	double RATELEARNING = 0.5;
	int ExTrain = 60000;
	int ExTest = 10000;		

	
	cout << "Datos de la Red" << endl;
	cout << "Estructura: ";
	for (int i = 0; i < sizes.size(); ++i)
	{
		cout << sizes[i] << ", ";
	}
	cout << endl;
	cout << "Velocidad de Aprendizaje: " << RATELEARNING << endl;
	cout << "Error min: " << ERROR << endl;
	cout << "Tipo Algoritmo: "Secuencial << endl;
	cout << "Ejemplos en entrenamiento: " << ExTrain	<< "	Ejemplos de prueba: " << ExTest << endl;
	cout << "Cantidad de epocas: " << EPOCAS << endl;

	vector<double> errores;
	printTime(errores = net.train_backpropagation(mnist.train_data, RATELEARNING, EPOCAS, ERROR, 60000));		
	//printTime(net.test_network(mnist.train_data, 10000));
	printTime(net.test_network(mnist.test_data, 10000));
	cout << "Error promedio alcanzado: " << errores[errores.size()-1] << endl;
	net.save("../models/pNet_0");
    return 0;
}
