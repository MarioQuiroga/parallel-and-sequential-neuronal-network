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
#include <time.h>
// Networks
//#include "../pNetwork/pNetwork.h"
#include "../common/loaderMnist.h"
#include "../common/ExampleChar.h"
#include "../sNetwork/Network.h"
//#include "../sNetwork/loaderMnist.h"
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
			int option;
			string s;
			readNum(socket, &option);
			switch(option)
			{
				case 0: readLine(socket, &s);
						writeLine(socket, s);
						close(socket);
						break;
				case 1: recnognition(socket);
						break;
				case 2: train(socket);
						break;
				case 3: stop(socket);
						break;
				case 4: saveModel(socket);
						break;
				default: close(socket);
			}
        }
    	catch(exception& e) 
    	{
    		cout << "Client disconnected." << endl;
        	cout << e.what() << endl;
    	}		
	}	

	void stop(int socket)
	{
		int pid;
		readNum(socket, &pid);
		cout << "Client disconected." << endl;
		kill(pid, SIGKILL);
		//waitpid(childProcess, &status, WNOHANG);
		close(socket);
	}

	void recnognition(int socket)
	{
		/*std::vector<double> input;
		readVector(socket, &input);
		Network net = Network(sizes);	
		net.load(PATH_TO_NETWORK);	
		int n = net.recogn(input);
		writeNum(socket, n);*/
	}

	void train(int socket)
	{
		MnistLoader mnist;	
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
		readVector(socket, &sizes);
		clock_t tStart, tEnd;
		// Loading data set
		int dataSources;
		readNum(socket, &dataSources);		
		if(dataSources==1){
			mnist = MnistLoader("../MNIST/train-images.idx3-ubyte", 
								"../MNIST/t10k-images.idx3-ubyte", 
								"../MNIST/train-labels.idx1-ubyte", 
								"../MNIST/t10k-labels.idx1-ubyte", 
								sizes[0],
								sizes[sizes.size()-1]);
		}else{
			//cout << sizes[0] << "!!" << sizes[sizes.size()-1] << endl;
			readInputData(socket, &mnist.train_data, examplesTrain, sizes[0], sizes[sizes.size()-1]);
			readInputData(socket, &mnist.test_data, examplesTest, sizes[0], sizes[sizes.size()]);
		}		
		
		// create and train network
		if(type_network==0)
		{
			Network net = Network(sizes);
			tm t_train;		
			tm t_test;
			vector<double> response;	
			getTime(response = net.train_backpropagation(mnist.train_data, rateLearning, epoch, error, examplesTrain), &t_train);							
			double test;
			getTime(test = net.test_network(mnist.test_data, examplesTest), &t_test);
			string s;
			double resp = response[response.size()-1];
			if(resp<=error)
			{
				s = "Ok";
			}
			else
			{
				s = "No-Ok";
			}
			writeLine(socket, s);	// Ok o No-Ok
			char buffer[20];			
			strftime(buffer, sizeof(buffer), "%H:%M:%S", &t_train);			
			string s_time(buffer);
			writeLine(socket, s_time);	// Train time
			int r = (int)response.size();
			writeNum(socket, &r); // Count epoch
			writeNum(socket, &response[response.size()-1]); // Error achieved
			writeNum(socket, &test); // Presicion
			writeVector(socket, response); // Vector errors
			writeModel(socket, net.getWeights(), net.getBias(), sizes);
		}
		else
		{
			if(type_network==1)
			{
				//Network_P net = Network_P(sizes);
				Network net = Network(sizes);
				tStart = clock();
				vector<double> response = net.train_backpropagation(mnist.train_data, rateLearning, epoch, error, examplesTrain);	
				tEnd = clock();
				clock_t train_time = tEnd-tStart;
				double test = net.test_network(mnist.test_data, examplesTest);
				string s;
				if(response[response.size()-1]<=error)
					s = "Ok";
				else
					s = "No-Ok";
				writeLine(socket, s);	// Ok o No-Ok
				writeNum(socket, &train_time);	// Train time
				int r = (int)response.size();
				writeNum(socket, &r); // Count epoch
				writeNum(socket, &response[response.size()-1]); // Error achieved
				writeNum(socket, &test); // Precision
				writeVector(socket, response); // Vector errors		
			}
		}
		
	}

	void readInputData(int socket, vector<ExampleChar> * v, int count, int inSize, int outSize){
		std::vector<double> fila;
		std::vector<ExampleChar> res;
		//cout << outSize << "-------------" << endl;
		for (int i = 0; i < count; i++)
		{
			readVector(socket, &fila);
			ExampleChar e = ExampleChar(inSize, outSize);
			e.label = (int) fila[0];
			//cout << (int) fila[0] << "||";
			for (int j = 1; j < inSize+1; j++)
			{
				e.input_data[j-1] = fila[j];
			}			

			for (int j = 0; j < outSize; j++)
			{

				//cout << "--" << j << "--" << e.label << endl;
				if(j==e.label)
				{
					e.output[j] = 1;
				}else
				{
					e.output[j] = 0;
				}
			}
			//cout << e.label << endl;
			/*for (int n = 0; n < outSize; n++)
			{
				cout << e.output[n] << "|";
			}*/
			/*cout << endl;
			for (int j = 0; j < 28; j++)
			{
				for (int k = 0; k < 28; k++)
				{
					if(e.input_data[j*28+k]==0)
					{
						std::cout << " ";
					}else
					{
						std::cout << "*";
					}							
				}
				std::cout << std::endl;
			}			
			std::cout << "----------------------------------" << std::endl;*/
			res.push_back(e);
		}
		//cout << endl;
		*v = res;

		
	}

	void saveModel(int socket)
	{
		string log;
		readLine(socket, &log);
		std::vector<double> model;
		readVector(socket, &model);
		ofstream file_log ("logs/", ios::binary);
		ofstream file_model ("models/", ios::binary);
	}
};
#endif
