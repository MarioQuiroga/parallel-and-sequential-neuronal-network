#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "common/loaderMnist.h"
#include "pNetwork/pNetwork.h"

int main()
{
	clock_t tStart, tEnd;
	//Constructor MnistLoader
	MnistLoader mnist = MnistLoader("../MNIST/train-images.idx3-ubyte", 
									"../MNIST/t10k-images.idx3-ubyte", 
									"../MNIST/train-labels.idx1-ubyte", 
									"../MNIST/t10k-labels.idx1-ubyte");
	
	//METODO PARA IMPRIMIR TODOS LOS DATOS: 0 IMPRIME DATOS DE ENTRENAMIENTO, 
	//										1 IMPRIME DATOS DE PRUEBA
	//mnist.print_data_set(0);	
	//cout << mnist.train_data.size() << endl;
	
	//CREO LA ESTRUCTURA DE LA RED
	vector<int> sizes;	
	sizes.push_back(784); sizes.push_back(100); sizes.push_back(50); sizes.push_back(15); sizes.push_back(10);			
	Network_P net = Network_P(sizes);	
	
	//net.mostrar_pesos();  
	//net.test_network(mnist.train_data, 50);
	int EPOCAS = 6000;
	double ERROR = 0.00000001;
	double RATELEARNING = 0.5;		
	
	tStart = clock();
	net.train_backpropagation(mnist.train_data, RATELEARNING, EPOCAS, ERROR, 600);		
	tEnd = clock();
	clock_t train_time = tEnd-tStart;
	cout << "Tiempo de entrenamiento en red paralela: " << train_time << endl;
	//net.test_network(mnist.test_data, 10000);
	//net.mostrar_output();

    return 0;
}
