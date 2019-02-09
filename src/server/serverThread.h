#ifndef serverThread_h
#define serverThread_h
#include <iostream>
#include <fstream>
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
#include <dirent.h>
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
				case 5: getModels(socket);
						break;
				default: close(socket);
			}
			close(socket);
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
	}

	void recnognition(int socket)
	{
		string name;
		readLine(socket, &name);
		std::vector<double> in;
		readVector(socket, &in);
		double * input = (double *) malloc(sizeof(double) * (in.size()-1));
		for (int i = 1; i < in.size(); i++)
		{
			input[i-1] = in[i];
		}
		int type_network;
		readNum(socket, &type_network);
		std::vector<int> sizes;
		ifstream file("../models/" + name);
		int layers;
		file >> layers;
		for (int i = 0; i < layers; i++)
		{
			int n;
			file >> n;
			sizes.push_back(n);
		}
		if(type_network==0)
		{ // SEQUENTIAL MODEL
			Network net = Network(sizes);	
			net.load("../models/" + name);	
			int n = net.recogn(input);
			writeNum(socket, &n);
		}else
		{
			if(type_network==1)
			{ // PARALLEL MODEL
				Network net = Network(sizes);
				//Network_P net = Network_P(sizes);	
				net.load("../models/" + name);	
				int n = net.recogn(input);
				writeNum(socket, &n);
			}
		}
		free(input);	
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
			mnist = MnistLoader("../MNIST/data/train-images.idx3-ubyte", 
								"../MNIST/data/t10k-images.idx3-ubyte", 
								"../MNIST/data/train-labels.idx1-ubyte", 
								"../MNIST/data/t10k-labels.idx1-ubyte", 
								sizes[0],
								sizes[sizes.size()-1]);
		}else{
			//cout << sizes[0] << "!!" << sizes[sizes.size()-1] << endl;
			readInputData(socket, &mnist.train_data, examplesTrain, sizes[0], sizes[sizes.size()-1]);
			readInputData(socket, &mnist.test_data, examplesTest, sizes[0], sizes[sizes.size()]);
		}		
		
		// create and train network
		if(type_network==0)
		{ // SEQUENTIAL MODEL
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
		{ // PARALEL MODEL
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
    	ifstream file_in("../logs/count");
    	int count; 
    	file_in >> count;
    	file_in.close();
    	ofstream file_out("../logs/count");
    	file_out << count + 1; 
    	cout << count << endl;
    	file_out.close();
		string name;
		readLine(socket, &name);
		std::vector<double> model;
		readVector(socket, &model);		
		string log;
		readLine(socket, &log);
		cout << log << endl;
		
		// openFiles
		cout << name << endl;
		string logs("../logs/");
		string models("../models/");
		logs = logs + name;
		models = models + name;
		cout << logs << endl;
		cout << models<< endl;
		std::ofstream file_log(logs, std::ofstream::out);
		std::ofstream file_model(models, std::ofstream::out);
		file_log << log;
		for (int i = 0; i < model.size(); i++)
		{
			file_model << model[i] << endl;
		}
		file_model.close();
		file_log.close();
	}
	void getModels(int socket)
	{
    	DIR *ID_Directorio_logs;
		dirent *Directorio_logs;
		ID_Directorio_logs = opendir("../logs");
		//Directorio_logs = readdir(ID_Directorio_logs);		

		DIR *dir;
		dirent *ent;
		//for(int i = 0; i < Directorio_logs->d_reclen; i++)
		int c = 0;
		std::vector<string> names;
		std::vector<string> logs;
		while (ent = readdir(ID_Directorio_logs))
		{	

			string name(ent->d_name);
			if(name.compare("..")!=0 & name.compare(".")!=0){
				names.push_back(name);
				string res_log("");
				string path("../logs/" + name);
				ifstream file_log(path);
				string linea;
				while(getline(file_log, linea))
				{
					res_log = res_log + linea + "\n";
				}		
				logs.push_back(res_log);	
			}				
		}
		int size = names.size();
		writeNum(socket, &size);
		for (int i = 0; i < names.size(); i++)
		{
//cout << names[i] << endl;
			writeLine(socket, names[i]);
			writeLine(socket, logs[i]);
		}		
		closedir (ID_Directorio_logs);
	}
};
#endif
