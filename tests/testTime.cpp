#include <stdio.h>
#include <iostream>
#include <time.h>
#include "../src/common/utilsTime.h"

#include <chrono>
#include <thread>

int main()
{
	tm t;
	printTime(std::this_thread::sleep_for(std::chrono::milliseconds(5000)));
	getTime(std::this_thread::sleep_for(std::chrono::milliseconds(5000)), &t);
	std::cout << "Tiempo total: "  
			  << t.tm_hour << ":"  
			  << t.tm_min << ":"  
			  << t.tm_sec 
			  << std::endl;
	return 0;
}