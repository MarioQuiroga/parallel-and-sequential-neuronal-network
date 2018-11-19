#include "serverThread.h"
#include <iostream>
#include <stdio.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <netinet/in.h>
#include "utilsTCP.h"

#define BACKLOG 10

using namespace std;

void sigchld_handler(int s)
{
    while(wait(NULL) > 0);
}

void thread_conection(int client, int id)
{
	serverThread thread = serverThread(client, id);
	thread.handlerConection();
}

class Server
{
	int countId = 0;		

public:
	Server(int port)
	{
		int listener = open_socket();		
		bind_to_port(listener, port);
		struct sigaction sa;
		if(listen(listener, BACKLOG)==-1)
		{
			cout << "Cola de mensajes llena" << endl;
			return;
		}

		cout << "Server Listener" << endl;
		
		int connect;
		while(1)
		{
			sa.sa_handler = sigchld_handler; // Eliminar procesos muertos
	        sigemptyset(&sa.sa_mask);
	        sa.sa_flags = SA_RESTART;
	        if (sigaction(SIGCHLD, &sa, NULL) == -1) {
	            perror("sigaction");
	            exit(1);
	        }
			struct sockaddr_storage client;
			unsigned int addres_size = sizeof(client);
			cout << "Esperando al cliente" << endl;
			// Create socket client			
			try
			{
				connect = accept(listener, (struct sockaddr*)&client, &addres_size);
				if(connect == -1)
				{
					cout << "No se pudo abrir socket secundario" << endl;
				}
				cout << "Atendiendo al cliente " << countId << endl;
				if (!fork()) 
				{ // Este es el proceso hijo
	                close(listener); // El hijo no necesita este descriptor
	                thread_conection(connect, countId);
	                close(connect);
	                exit(0);
            	}
            	countId++;            	
            	close(connect);  // El proceso padre no lo necesita				
			}
			catch(exception& e) 
			{
				cout << "Problemas en connect." << endl;
		    	cout << e.what() << endl;
			}					
		}
	}	
};

// To test 
int main()
{
	Server s = Server(9000);
	return 0;
}