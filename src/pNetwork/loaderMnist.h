#ifndef loaderMnist_h
#define loaderMnist_h
#include <iostream>
#include <fstream>
#include <vector>

// BIG-Endian to LITTLE-Endian byte swap
#define swap16(n) (((n&0xFF00)>>8)|((n&0x00FF)<<8))
#define swap32(n) ((swap16((n&0xFFFF0000)>>16))|((swap16(n&0x0000FFFF))<<16))

typedef unsigned char  byte;

struct ExampleChar {
    double input_data[784];          // Store the 784 (28x28) pixel color values (0-255) of the digit-image
    double output[10];             // Store the expected output (e.g: label 5 / output 0,0,0,0,0,1,0,0,0,0)
    int label;                              // Store the handwritten digit in number form
    ExampleChar() : label(0) {}
};


void loadData(std::fstream * file_images, std::fstream * file_labels, std::vector<ExampleChar> * data)
{
	//VERIFICANDO LA INTEGRIDAD DE LOS ARCHIVOS
	int magicNum_images = 0, magicNum_labels = 0;
	file_images->read((char*)&magicNum_images, 4);									
	file_labels->read((char*)&magicNum_labels, 4);						
	magicNum_images = swap32(magicNum_images);
	magicNum_labels = swap32(magicNum_labels);
	if(magicNum_images==2051 && magicNum_labels==2049){
		int itemCount_images = 0, itemCount_labels = 0;
		int row_count = 0, col_count = 0;
		file_labels->read((char*)&itemCount_labels, 4);					
		file_images->read((char*)&itemCount_images, 4);			
		itemCount_labels = swap32(itemCount_labels);
		itemCount_images = swap32(itemCount_images);
		//std::cout <<  itemCount_images << std::endl;
		//std::cout << itemCount_labels << std::endl;
		file_images->read((char*)&row_count, 4);			
		file_images->read((char*)&col_count, 4);				
		row_count = swap32(row_count);
		col_count = swap32(col_count);
		//std::cout << row_count;
		//std::cout << " x ";
		//std::cout << col_count << std::endl; 
		
		for (int i = 0; i < itemCount_images; i++) {
			ExampleChar tmpchar = ExampleChar();
			byte label;
			for(int r = 0; r < (row_count * col_count); r++) {
				byte pixel = 1;						
				double p;
				// read one byte (0-255 color value of the pixel)
				file_images->read((char*)&pixel, sizeof(pixel));				
				p = (double) pixel;
				p = p/1000;
				tmpchar.input_data[r] = p;
				//tmpchar.input_data.push_back(p);						
			}			
			file_labels->read((char*)&label, 1);
			tmpchar.label = (int) label;
			for(int o=0; o<10;o++){
				if(o==tmpchar.label){
					tmpchar.output[o] = 1;
				}else{
					tmpchar.output[o] = 0;
				}						
			}						
			data->push_back(tmpchar);
			
		}	
		
	}else{
		if(magicNum_images!=2051) std::cout << "Error, numero magico de imagenes no valido. Verifique la integridad del archivo." << std::endl;
		if(magicNum_labels!=2049) std::cout << "Error, numero magico de etiquetas no valido. Verifique la integridad del archivo." << std::endl;				
	}								
}
	
class MnistLoader
{
	public:
		//Data
		std::vector<ExampleChar> train_data;
		std::vector<ExampleChar> test_data;
		
		//Constructor
		MnistLoader(const std::string path_train_images, 
					const std::string path_test_images, 
					const std::string path_train_labels, 
					const std::string path_test_labels)
		{
			std::cout << "Cargando DataSet MNIST..." << std::endl;
			std::fstream file_train_images (path_train_images, std::ifstream::in | std::ifstream::binary);
			std::fstream file_test_images (path_test_images, std::ifstream::in | std::ifstream::binary);
			std::fstream file_train_labels (path_train_labels, std::ifstream::in | std::ifstream::binary);
			std::fstream file_test_labels (path_test_labels, std::ifstream::in | std::ifstream::binary);
			
			train_data = std::vector<ExampleChar>();
			test_data = std::vector<ExampleChar>();
			
			if(file_train_images.is_open()&&file_train_labels.is_open()){
				loadData(&file_train_images, &file_train_labels, &train_data);			
				file_train_images.close();
				file_train_labels.close();
				
			}else{
				std::cout << "Error al abrir los archivos de Entrenamiento. Verifique los paths" << std::endl;
			}		
			
			if (file_test_labels.is_open()&&file_test_images.is_open()){
				loadData(&file_test_images, &file_test_labels, &test_data);			
				file_test_images.close();
				file_test_labels.close();
			}else{
				std::cout << "Error al abrir los archivos de Prueba. Verifique los paths" << std::endl;
			}
			
		}
		
		void print_data_set(int set, int count)
		{
			if(set==0){ // IMPRIMO DATOS DE ENTRENAMIENTO
				
				for(int i=0; i<count; i++)
				{	
					std::cout << "Label: " ;
					std::cout << train_data[i].label << std::endl;
					std::cout << "Out: ";
					for (int o=0; o<10; o++){
						std::cout << train_data[i].output[o];
					}
					std::cout << std::endl;					
					for(int j=0; j<28;j++)
					{
						for(int k=0; k<28; k++)
						{
							if(train_data[i].input_data[j*28+k]==0){
								std::cout << " ";
							}else{
								std::cout << "*";
							}							
						}
						std::cout << std::endl;
					}						
					std::cout << "----------------------------------" << std::endl;
					
				}
			}else{
				if(set==1){ // IMPRIMO DATOS DE PRUEBA
					for(int i=0; i<count; i++)
					{	
						std::cout << "Label: " ;
						std::cout << test_data[i].label << std::endl;
						std::cout << "Out: ";
						for (int o=0; o<10; o++){
							std::cout << test_data[i].output[o];
						}
						std::cout << std::endl;					
						for(int j=0; j<28;j++)
						{
							for(int k=0; k<28; k++)
							{
								if(test_data[i].input_data[j*28+k]==0){
									std::cout << " ";
								}else{
									std::cout << "*";
								}								
							}
							std::cout << std::endl;
						}						
						std::cout << "----------------------------------" << std::endl;
					}
				}	
			}
		}
};
#endif