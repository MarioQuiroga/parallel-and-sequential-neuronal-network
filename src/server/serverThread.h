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
#include <vector>

// Networks
#include "../pNetwork/pNetwork.h"
#include "../common/loaderMnist.h"
#include "../sNetwork/Network.h"
#include "../sNetwork/loaderMnist.h"
#include "../common/utilsCommon.h"

#define PATH_TO_NETWORK ""

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
        	/*std::string s;
			readLine(socket, &s);
			cout << "Recibido: " << s << endl;
			std::string sNew("Este es el nuevo string");
			writeLine(socket, sNew);*/
			int option;
			readNum(socket, &option);
			switch(option)
			{
				case 0: recnognition(socket);
						break;
				case 1: train(socket);
						break;
				default: close(socket);
			}
        }
    	catch(exception& e) 
    	{
    		cout << "Cliente desconectado." << endl;
        	cout << e.what() << endl;
    	}		
	}	

	void recnognition(int socket)
	{
		std::vector<double> input;
		readVector(socket, &input);
		Network net = Network(sizes);	
		net.load(PATH_TO_NETWORK);	
		int n = net.recogn(input);
		writeNum(socket, n);
	}

	void train(int socket)
	{
		// reading params
		int type_network;
		readNum(socket, &type_network); 
		double error;
		readNum(socket, &error);
		double rateLearning;
		readNum(socket, &rateLearning);
		int epoch;
		readNum(socket, &epoch);
		int examplesTrain;
		readNum(socket, &examplesTrain);
		int examplesTest;
		readNum(socket, &examplesTest);

		// reading struct network
		std::vector<int> sizes;
		int numLayers;
		readNum(socket, &numLayers);
		for (int i = 0; i < numLayers; ++i)
		{
			int size;
			readNum(socket, &size);
			sizes.push_back(size);
		}

		clock_t tStart, tEnd;
		// Loading data set
		MnistLoader mnist = MnistLoader("../MNIST/train-images.idx3-ubyte", 
										"../MNIST/t10k-images.idx3-ubyte", 
										"../MNIST/train-labels.idx1-ubyte", 
										"../MNIST/t10k-labels.idx1-ubyte");
		// create and train network
		if(type_network==0)
		{
			Network net = Network(sizes);
			tStart = clock();
			vector<double> response = net.train_backpropagation(mnist.train_data, rateLearning, epoch, error, examplesTrain);	
			tEnd = clock();
			clock_t train_time = tEnd-tStart;
			double test = net.test_network(mnist.train_data, examplesTest);
			string s;
			if(response[response.size()-1]=<error)
				s = "Ok";
			else
				string s("No-Ok");
			writeLine(socket, s);	// Ok o No-Ok
			writeNum(socket, train_time);	// Train time
			writeNum(socket, response.size()); // Count epoch
			writeNum(socket, response[response.size()-1]); // Error achieved
			writeNum(socket, test); // Presicion
			writeVector(socket, response); // Vector errors
		}
		else
		{
			if(type_network==1)
			{
				Network_P net = Network_P(sizes);
				tStart = clock();
				vector<double> response = net.train_backpropagation(mnist.train_data, rateLearning, epoch, error, examplesTrain);	
				tEnd = clock();
				clock_t train_time = tEnd-tStart;
				double test = net.test_network(mnist.train_data, examplesTest);
				string s;
				if(response[response.size()-1]=<error)
					s = "Ok";
				else
					string s("No-Ok");
				writeLine(socket, s);	// Ok o No-Ok
				writeNum(socket, train_time);	// Train time
				writeNum(socket, response.size()); // Count epoch
				writeNum(socket, response[response.size()-1]); // Error achieved
				writeNum(socket, test); // Presicion
				writeVector(socket, response); // Vector errors		
			}
		}
	}
};
#endif