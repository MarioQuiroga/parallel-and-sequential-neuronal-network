#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "common/loaderMnist.h"
#include "pNetwork/pNetwork.h"
#include "sNetwork/Network.h"
#include "common/utilsTime.h"

int main()
{
	//Constructor MnistLoader
	MnistLoader mnist = MnistLoader("../MNIST/data/train-images.idx3-ubyte", 
									"../MNIST/data/t10k-images.idx3-ubyte", 
									"../MNIST/data/train-labels.idx1-ubyte", 
									"../MNIST/data/t10k-labels.idx1-ubyte", 784, 10);

	int typeNet = 1; // 0 PARALLEL, 1 SEQUENTIAL
	vector<int> sizes;	
	sizes.push_back(784); sizes.push_back(500); sizes.push_back(300); sizes.push_back(100); sizes.push_back(100);  sizes.push_back(10);			
	int EPOCAS = 10;
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
	cout << "Tipo Algoritmo: "<< typeNet << endl;
	cout << "Ejemplos en entrenamiento: " << ExTrain	<< "	Ejemplos de prueba: " << ExTest << endl;
	cout << "Cantidad de epocas: " << EPOCAS << endl;
	vector<double> errores;
	if(typeNet == 0)
	{
		Network_P net = Network_P(sizes);		
		printTime(errores = net.train_backpropagation(mnist.train_data, RATELEARNING, EPOCAS, ERROR, ExTrain));		
		printTime(net.test_network(mnist.test_data, ExTest));
		cout << "Error promedio alcanzado: " << errores[errores.size()-1] << endl;
		net.save("../models/pNet_3");
	}
	else
	{
		Network net = Network(sizes);		
		printTime(errores = net.train_backpropagation(mnist.train_data, RATELEARNING, EPOCAS, ERROR, ExTrain));		
		printTime(net.test_network(mnist.test_data, ExTest));
		cout << "Error promedio alcanzado: " << errores[errores.size()-1] << endl;
		net.save("../models/sNet_3");
	}	
    return 0;
}
