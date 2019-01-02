#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "sNetwork/loaderMnist.h"
#include "sNetwork/Network.h"
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
	//mnist.print_data_set(1);	
	//cout << mnist.train_data.size() << endl;
	
	//CREO LA ESTRUCTURA DE LA RED
	vector<int> sizes;	
	sizes.push_back(784); sizes.push_back(512); sizes.push_back(100); sizes.push_back(50); sizes.push_back(10);			
	//sizes.push_back(3); sizes.push_back(5); sizes.push_back(2);
	Network net = Network(sizes);	
	
	//net.mostrar_pesos();  
	//net.mostrar_output();	
	//net.test_network(mnist.train_data, 50);
	int EPOCAS = 6000;
	double ERROR = 0.0001;
	double RATELEARNING = 30;		

	printTime(net.train_backpropagation(mnist.train_data, RATELEARNING, EPOCAS, ERROR, 20));	
	printTime(net.test_network(mnist.train_data, 10));
	//net.mostrar_output();
	net.save("pNet");
    return 0;
}
