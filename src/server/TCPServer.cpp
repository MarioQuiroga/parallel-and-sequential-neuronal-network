#include "serverThread.h"
#include <stdio.h>
#include <string.h>
#include <winsock.h>
//#include <arpa/inet.h>

using namespace std;

/*void sigchld_handler(int s)
{
    while(wait(NULL) > 0);
}*/

int open_socket()
{
	int s = socket(AF_INET, SOCK_STREAM, 0);
	if(s == -1)
	{
		cout << "Error en listener al abrir el socket" << endl
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
	if(setsockopt(socket, SOL_SOCKET, SO_REUSADOR, (char*)&reuse, sizeof(int)) == -1)
	{
		Error("No es posible reusar el socket\n");
	}
	/*BIND*/
	int c = bind(socket, (struct sockaddr*) &name, sizeof(name));
	if(c==-1)
	{
		perror("No es posible enlazar al puerto, direccion en uso\n");
	}
}

class Server
{


public:
	Server(int port)
	{
		struct sigaction sa;		
		int listener = open_socket();		
		bind_to_port(listener, port);
		if(listen(listener, 10)==-1)
		{
			cout << "Cola de mensajes llena" << endl;
			return;
		}
		/*sa.sa_handler = sigchld_handler; // Eliminar procesos muertos
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        if (sigaction(SIGCHLD, &sa, NULL) == -1) 
        {
            perror("sigaction");
            exit(1);
        }*/

		cout << "Server Listener" << endl;
		while(1)
		{
			struct sockaddr_storage client;
			unsigned int addres_size = sizeof(client);
			cout << "Esperando al cliente" << endl;
			// Creo un socket secundario
			int connect = accept(listener, (struct sockaddr*)&client, &addres_size);
			if(connect == -1)
			{
				cout << "No se pudo conectar socket secundario" << endl;
			}
			cout << "Atendiendo al cliente" << endl;
			/*if (!fork()) { // Este es el proceso hijo
                close(listener); // El hijo no necesita este descriptor
                if (send(connect, "Hello, world!\n", 14, 0) == -1)
                    perror("send");
                close(connect);
                exit(0);
            }
            close(new_fd);  // El proceso padre no lo necesita*/
			
			char * msg = "Mensaje al cliente";
			send(connect, msg, strlen(msg), 0);
			msg = NULL; // o 0 ver?
			close(connect);
		}
	}
	~Server();
	
};





