#include <stdio.h>
#include <cmath>
#include <vector>

using namespace std; 

/**
*	FUNCIÓN SIGMOIDE Y SU DERIVADA
*
*/
double sigmoid(double x)
{
	return 1/(1+exp(-x));
}
double sigmoid_prima(double x)
{
	return (sigmoid(x) * (1 - sigmoid(x)));
}

/**
*	DEVUELVE LA POSICIÓN DEL MAYOR ELEMENTO DE UN VECTOR
*
*/	
int index_max(vector<double> entrada)
{
	int indice = 0;
	double max = entrada[0];
	for(int i=1;i<entrada.size();i++)
	{
		if(entrada[i] > max)
		{
			max = entrada[i];
			indice = i;
		}
	}
	return indice;	
}
