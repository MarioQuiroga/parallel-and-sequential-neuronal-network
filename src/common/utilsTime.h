#ifndef utilsTime_h
#define utilsTime_h
#include <time.h>

#define getTime(task, pointer_tm){ time_t first, second; first = time(NULL); task; second = time(NULL);	tm res_time = getTm(difftime(second, first)); *pointer_tm = res_time; }
#define printTime(task){ time_t first, second; first = time(NULL); task; second = time(NULL);	tm res_time = getTm(difftime(second, first)); std::cout << "Tiempo total: " << res_time.tm_hour << ":"  << res_time.tm_min << ":" << res_time.tm_sec << std::endl; }

tm getTm(int tsegundos)
{
	tm response;
	response.tm_hour = (tsegundos / 3600);
	response.tm_min = ((tsegundos-response.tm_hour*3600)/60);
	response.tm_sec = tsegundos-(response.tm_hour*3600+response.tm_min*60);
	return response;
}

#endif