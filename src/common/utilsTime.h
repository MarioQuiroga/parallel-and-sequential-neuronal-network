#ifndef utilsTime_h
#define utilsTime_h
#include <time.h>

tm getTm(int tsegundos)
{
	tm response;
	response.tm_hour = (tsegundos / 3600);
	response.tm_min = ((tsegundos-response.tm_hour*3600)/60);
	response.tm_sec = tsegundos-(response.tm_hour*3600+response.tm_min*60);
	return response;
}

#endif