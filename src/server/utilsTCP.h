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
#include <vector>

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

template<typename T>
T readNum(int socket, T * n)
{
	if (recv(socket, n, sizeof(T), 0) == -1) 
	{
        perror("Error in recv Num");
        exit(1);
    }
}

template<typename T>
T writeNum(int socket, T * n)
{
	if (send(socket, n, sizeof(T), 0) == -1) 
	{
        perror("Error in send Num");
        exit(1);
    }
}

template<typename T>
void writeVector(int socket, std::vector<T> input)
{
	int size = (int)input.size();
	writeNum(socket, &size);
	for (int i = 0; i < input.size(); ++i)
	{
		writeNum(socket, &input[i]);
	}
}

template<typename T>
void readVector(int socket, std::vector<T> * input)
{
	std::vector<T> res;
	T c;
	int n;
	readNum(socket, &n);
	for (int i = 0; i < n; ++i)
	{
		readNum(socket, &c);
		res.push_back(c);
	}
	*input = res;
}

void readLine(int socket, std::string * line)
{
	//cout << "Recibiendo..."<< endl;
	int n;
	readNum(socket, &n);
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
	writeNum(socket, &n);
	if (send(socket, c, sizeof(char)*n, 0) == -1)
		perror("Error in send");
}

void writeModel(int socket, vector<vector<vector<double>>> w, vector<vector<double>> b, vector<int> sizes)
{
	std::vector<double> response;
	response.push_back(sizes.size());	
	for (int i = 0; i < sizes.size(); i++)
	{
		response.push_back(sizes[i]);
	}
	for (int i = 0; i < w.size(); i++)
	{
		for (int j = 0; j < w[i].size(); j++)
		{
			for (int k = 0; k < w[i][j].size(); k++)
			{
				response.push_back(w[i][j][k]);
			}
		}
	}
	for (int j = 0; j < b.size(); j++)
	{
		for (int k = 0; k < b[j].size(); k++)
		{
			response.push_back(b[j][k]);
		}
	}	
	writeVector(socket, response);
}

#endif