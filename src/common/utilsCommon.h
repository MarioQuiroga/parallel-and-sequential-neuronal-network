#ifndef utilsCommon_h
#define utilsCommon_h
#include <stdio.h>
#include <cmath>
#include <vector>

using namespace std; 

/*struct responseRec(){

};*/

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
#endif



