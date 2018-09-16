#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "loaderMnist.h"
#include "Network.h"

int main()
{
	//Constructor MnistLoader
	MnistLoader mnist = MnistLoader("MNIST/train-images.idx3-ubyte", 
									"MNIST/t10k-images.idx3-ubyte", 
									"MNIST/train-labels.idx1-ubyte", 
									"MNIST/t10k-labels.idx1-ubyte");
	
	//METODO PARA IMPRIMIR TODOS LOS DATOS 0 IMPRIME DATOS DE ENTRENAMIENTO 1 IMPRIME DATOS DE PRUEBA
	//mnist.print_data_set(0);	
	//cout << mnist.train_data.size() << endl;
	
	//CREO LA ESTRUCTURA DE LA RED
	vector<int> sizes;	
	sizes.push_back(784); sizes.push_back(100); sizes.push_back(50); sizes.push_back(15); sizes.push_back(10);			
	Network net = Network(sizes);	
	
	//net.mostrar_pesos();  net.mostrar_output();	
	//net.test_network(mnist.train_data, 50);
	int EPOCAS = 6000;
	double ERROR = 0.001;
	double RATELEARNING = 0.5;		
	
	net.train_backpropagation(mnist.train_data, RATELEARNING, EPOCAS, ERROR, 50);	
	
	net.test_network(mnist.train_data, 50);
	//net.mostrar_output();

    return 0;
}