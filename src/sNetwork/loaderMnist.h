#ifndef loaderMnist_h
#define loaderMnist_h
#include <iostream>
#include <fstream>
#include <vector>

// BIG-Endian to LITTLE-Endian byte swap
#define swap16(n) (((n&0xFF00)>>8)|((n&0x00FF)<<8))
#define swap32(n) ((swap16((n&0xFFFF0000)>>16))|((swap16(n&0x0000FFFF))<<16))

struct ExampleChar {
    std::vector<double> input_data;          // Store the 784 (28x28) pixel color values (0-255) of the digit-image
    std::vector<double> output;             // Store the expected output (e.g: label 5 / output 0,0,0,0,0,1,0,0,0,0)
    int label;                              // Store the handwritten digit in number form
    ExampleChar() : output(10, 0), label(-1) {}
};

std::ostream& operator<<(std::ostream& os, const ExampleChar& ex) {
	os << "Label: " << ex.label << "\n";
	os << "Out: ";
	for (int o = 0; o < ex.output.size() - 1; o++)
		os << ex.output[o] << ", ";
	os << ex.output.back() << "\n";

	for(int j = 0; j < ex.input_data.size()-1;j++)
		os << ex.input_data[j] << " ";
	os << ex.input_data.back();
	return os;
}

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
			for(int r = 0; r < (row_count * col_count); r++) {
				unsigned char pixel = 1;
				// read one byte (0-255 color value of the pixel)
				file_images->read((char*)&pixel, sizeof(pixel));
				tmpchar.input_data.push_back(static_cast<double>(pixel) / 255.0);
			}
			unsigned char buf;
			file_labels->read((char*)&buf, sizeof(buf));
			tmpchar.label = buf;
			tmpchar.output[tmpchar.label] = 1;
			data->push_back(tmpchar);
		}
	}
	else {
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
		
		void print_data_set(int set, int start_idx=0, int end_idx=-1)
		{
			auto& data = set == 0 ? train_data : test_data;
			for (int i = start_idx; i != end_idx && i < data.size(); ++i)
				std::cout << data[i] << "\n--------------------" << std::endl;
		}
};
#endif
