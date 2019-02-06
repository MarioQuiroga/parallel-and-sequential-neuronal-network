#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>

// BIG-Endian to LITTLE-Endian byte swap
#define swap16(n) (((n&0xFF00)>>8)|((n&0x00FF)<<8))
#define swap32(n) ((swap16((n&0xFFFF0000)>>16))|((swap16(n&0x0000FFFF))<<16))

typedef unsigned char  byte;
using namespace std;
void loadData(std::fstream * file_images, std::fstream * file_labels, std::ofstream * file_csv, int inSize, int outSize)
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
		byte label;
		*file_csv << "label";
		for (int i = 0; i < inSize; ++i)
		{
			*file_csv << "," << "pixel" << i;
		}
		*file_csv << endl;		
		for (int i = 0; i < itemCount_images; i++) {			
			file_labels->read((char*)&label, 1);
			*file_csv << (int) label;			
			for(int r = 0; r < (row_count * col_count); r++) {
				byte pixel;						
				double p;
				// read one byte (0-255 color value of the pixel)
				file_images->read((char*)&pixel, sizeof(pixel));				
				p = (double) pixel/1000;
				*file_csv << "," << p;
				//tmpchar.input_data[r] = p;
				//tmpchar.input_data.push_back(p);						
			}
			*file_csv << endl;			
		}	
		
	}else{
		if(magicNum_images!=2051) std::cout << "Error, numero magico de imagenes no valido. Verifique la integridad del archivo." << std::endl;
		if(magicNum_labels!=2049) std::cout << "Error, numero magico de etiquetas no valido. Verifique la integridad del archivo." << std::endl;				
	}								
}
	

void mnistLoader(const std::string path_train_images, 
				 const std::string path_test_images, 
				 const std::string path_train_labels, 
				 const std::string path_test_labels,
				 const std::string path_train_csv,
				 const std::string path_test_csv, int inSize, int outSize)
{
	std::fstream file_train_images (path_train_images, std::ifstream::in | std::ifstream::binary);
	std::fstream file_test_images (path_test_images, std::ifstream::in | std::ifstream::binary);
	std::fstream file_train_labels (path_train_labels, std::ifstream::in | std::ifstream::binary);
	std::fstream file_test_labels (path_test_labels, std::ifstream::in | std::ifstream::binary);
	std::ofstream train_csv(path_train_csv, std::ofstream::out);
	std::ofstream test_csv(path_test_csv, std::ofstream::out);	
	
	if(file_train_images.is_open()&&file_train_labels.is_open()){
		loadData(&file_train_images, &file_train_labels, &train_csv, inSize, outSize);			
		file_train_images.close();
		file_train_labels.close();
		train_csv.close();
		
	}else{
		std::cout << "Error al abrir los archivos de Entrenamiento. Verifique los paths" << std::endl;
	}		
	
	if (file_test_labels.is_open()&&file_test_images.is_open()){
		loadData(&file_test_images, &file_test_labels, &test_csv, inSize, outSize);			
		file_test_images.close();
		file_test_labels.close();
		test_csv.close();
	}else{
		std::cout << "Error al abrir los archivos de Prueba. Verifique los paths" << std::endl;
	}
	
}


int main(int argc, char const *argv[])
{
	mnistLoader("train-images.idx3-ubyte", 
				"t10k-images.idx3-ubyte", 
				"train-labels.idx1-ubyte", 
				"t10k-labels.idx1-ubyte",
				"mnist-train.csv",
				"mnist-test.csv",
				784,
				10);
	return 0;
}