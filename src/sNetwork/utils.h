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
