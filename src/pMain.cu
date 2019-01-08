#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "pNetwork/loaderMnist.h"
#include "pNetwork/pNetwork.h"
#include "common/utilsTime.h"

int main()
{
	//Constructor MnistLoader
	MnistLoader mnist = MnistLoader("../MNIST/train-images.idx3-ubyte", 
									"../MNIST/t10k-images.idx3-ubyte", 
									"../MNIST/train-labels.idx1-ubyte", 
									"../MNIST/t10k-labels.idx1-ubyte");
	
	//METODO PARA IMPRIMIR TODOS LOS DATOS: 0 IMPRIME DATOS DE ENTRENAMIENTO, 
	//										1 IMPRIME DATOS DE PRUEBA
	//mnist.print_data_set(0, 10);	
	//cout << mnist.train_data.size() << endl;
	
	//CREO LA ESTRUCTURA DE LA RED
	vector<int> sizes;	
	sizes.push_back(784); sizes.push_back(100); sizes.push_back(15); sizes.push_back(10);			
	//sizes.push_back(3); sizes.push_back(5); sizes.push_back(2);;			
	Network_P net = Network_P(sizes);	
	//Network net = Network(sizes);	
	//net.mostrar_pesos();
	
	//Network_P net1 = Network_P(sizes);	

	//net.load("pNet");

	//net1.mostrar_pesos();
	//net.mostrar_pesos(); 
	//printf("Error: %s \n", cudaGetErrorName(cudaGetLastError()));
	//net.test_network(mnist.train_data, 50);
	int EPOCAS = 6000;
	double ERROR = 0.001;
	double RATELEARNING = 0.5;	

	time_t first, second;

	first = time(NULL);  
	printTime(vector<double> erroes = net.train_backpropagation(mnist.train_data, RATELEARNING, EPOCAS, ERROR, 60000));		
	second = time(NULL);

	cout << "Tiempo entrenamiento: " << difftime(second, first) << " segundos\n";
	tm train_time = getTm(difftime(second, first));
	cout << "tm_struct: " << train_time.tm_hour << ":" << train_time.tm_min << ":" << train_time.tm_sec << endl;
	printTime(net.test_network(mnist.train_data, 10000));
	net.save("pNet");
	//net.mostrar_output();*/
	printTime(net.test_network(mnist.test_data, 10000));
    return 0;
}
