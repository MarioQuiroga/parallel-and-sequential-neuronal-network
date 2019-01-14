#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
//#include "loaderMnist.h"
#include "../src/sNetwork/Network.h"


int main(){

	//CREO LA ESTRUCTURA DE LA RED
	vector<int> sizes;	
	sizes.push_back(2);	sizes.push_back(5); sizes.push_back(2);			
	Network net = Network(sizes);	
	
	//net.mostrar_pesos();

	//net.save("testFileNet");

	//Network net1 = Network(sizes);	
	//Network net1 = Network(sizes);	

	//net1.load("testFileNet");

	net.mostrar_pesos();
	
	//DATOS PARA EL ENTRENAMIENTO
	/*vector<ExampleChar> datos;	
	ExampleChar tmpchar1 = ExampleChar();
	ExampleChar tmpchar2 = ExampleChar();
	ExampleChar tmpchar3 = ExampleChar();
	ExampleChar tmpchar4 = ExampleChar();
	
	tmpchar1.input_data.push_back(1); tmpchar1.input_data.push_back(1);	
	tmpchar1.output.push_back(1); tmpchar1.output.push_back(0);
	tmpchar1.label=0;
	datos.push_back(tmpchar1);
	
	tmpchar2.input_data.push_back(0); tmpchar2.input_data.push_back(0);	
	tmpchar2.output.push_back(1); tmpchar2.output.push_back(0);
	tmpchar2.label=0;
	datos.push_back(tmpchar2);
	
	tmpchar3.input_data.push_back(1); tmpchar3.input_data.push_back(0);	
	tmpchar3.output.push_back(0); tmpchar3.output.push_back(1);
	tmpchar3.label=1;
	datos.push_back(tmpchar3);
	
	tmpchar4.input_data.push_back(0); tmpchar4.input_data.push_back(1);	
	tmpchar4.output.push_back(0); tmpchar4.output.push_back(1);
	tmpchar4.label=1;
	datos.push_back(tmpchar4);
	
	//net.test_network(datos, 4);
	//PARAMETROS PARA EL ENTRENAMIENTO
	int EPOCAS = 3000;
	double ERROR = 0.001;
	double RATELEARNING = 0.5;
	net.train_backpropagation(datos, RATELEARNING, EPOCAS, ERROR, 4);	
	
	net.test_network(datos, 4);*/

    return 0;
	
}