#ifndef serverThread_h
#define serverThread_h
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include "utilsTCP.h"

using namespace std;

class serverThread
{
	int id; 
	int socket;
	
public:
	serverThread(int conection, int idt)
	{
		id = idt;
		socket = conection;
	}

	void handlerConection()
	{
		try 
		{
        	std::string s;
			readLine(socket, &s);
			cout << "Recibido: " << s << endl;
			//convertir a mayuscula para pruebas
			std::string sNew("Este es el nuevo string");
			writeLine(socket, sNew);
        }
    	catch(exception& e) 
    	{
    		cout << "Cliente desconectado." << endl;
        	cout << e.what() << endl;
    	}		
	}	
};
#endif