#ifndef utilsTCP_h
#define utilsTCP_h
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

using namespace std;

int open_socket()
{
	int s = socket(PF_INET, SOCK_STREAM, 0);
	if(s == -1)
	{
		cout << "Error en listener al abrir el socket" << endl;
	}
	return s;
}

void bind_to_port(int socket, int port)
{
	struct sockaddr_in name;
	name.sin_family = AF_INET;
	name.sin_port   = (in_port_t)htons(port);
	name.sin_addr.s_addr = htonl(INADDR_ANY);
	/* Avoid probles reusing the port */
	int reuse = 1;
	if(setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(int)) == -1)
	{
		perror("No es posible reusar el socket\n");
	}
	/* BIND */
	int c = bind(socket, (struct sockaddr*) &name, sizeof(name));
	if(c==-1)
	{
		perror("No es posible enlazar al puerto, direccion en uso\n");
	}
}

void readLine(int socket, std::string * line)
{
	//cout << "Recibiendo..."<< endl;
	int n;
	if (recv(socket, &n, sizeof(int), 0) == -1) 
	{
        perror("Error in recv");
        exit(1);
    }
    char * c = new char[n];
    if (recv(socket, c, n, 0) == -1) 
	{
        perror("Error in recv");
        //exit(1);
    }
	*line = std::string(c);
}

void writeLine(int socket, std::string line)
{
	char * c = new char[line.length()];
	strcpy(c, line.c_str());
	int n = line.length();
	if (send(socket, &n, sizeof(int), 0) == -1)
		perror("Error in send");

	if (send(socket, c, sizeof(char)*n, 0) == -1)
		perror("Error in send");
}

#endif