#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include "../src/pNetwork/loaderMnist.h"
#include "../src/pNetwork/pNetwork.h"
#include "../src/common/utilsTime.h"


int main()
{
	//CREO LA ESTRUCTURA DE LA RED
	vector<int> sizes;	
	sizes.push_back(2);	sizes.push_back(5); sizes.push_back(2);			
	Network_P net = Network_P(sizes);	
	
	net.mostrar_pesos();

	//net.save("testFileNet");

	//Network net1 = Network(sizes);	
	//Network_P net1 = Network_P(sizes);	

	//net1.load("testFileNet");

	//net1.mostrar_pesos();
	
	//DATOS PARA EL ENTRENAMIENTO
	/*vector<ExampleChar> datos = std::vector<ExampleChar>();	
	ExampleChar tmpchar1 = ExampleChar();
	ExampleChar tmpchar2 = ExampleChar();
	ExampleChar tmpchar3 = ExampleChar();
	ExampleChar tmpchar4 = ExampleChar();
	
	tmpchar1.input_data[0]=1; tmpchar1.input_data[1]=1;	
	tmpchar1.output[0]=1; tmpchar1.output[1]=0;
	tmpchar1.label=0;
	datos.push_back(tmpchar1);
	
	tmpchar2.input_data[0]=0; tmpchar2.input_data[1]=0;		
	tmpchar2.output[0]=1; tmpchar2.output[1]=0;
	tmpchar2.label=0;
	datos.push_back(tmpchar2);
	
	tmpchar3.input_data[0]=1; tmpchar3.input_data[1]=0;	
	tmpchar3.output[0]=0; tmpchar3.output[1]=1;	
	tmpchar3.label=1;
	datos.push_back(tmpchar3);
	
	tmpchar4.input_data[0]=0; tmpchar4.input_data[1]=1;	
	tmpchar4.output[0]=0; tmpchar4.output[1]=1;		
	tmpchar4.label=1;
	datos.push_back(tmpchar4);
	
	//net.test_network(datos, 4);
	//PARAMETROS PARA EL ENTRENAMIENTO
	int EPOCAS = 60000;
	double ERROR = 0.001;
	double RATELEARNING = 0.5;
	printTime(net.train_backpropagation(datos, RATELEARNING, EPOCAS, ERROR, 4));	
	
	net.test_network(datos, 4);
	net.save("xor");*/

    return 0;
	
}