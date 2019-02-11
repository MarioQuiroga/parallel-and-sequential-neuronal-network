#ifndef exampleChar_h
#define exampleChar_h

class ExampleChar
{
public:
	double * input_data;
	double * output;
	int label;

	__host__ __device__ ExampleChar(int inSize, int outSize)
	{
		input_data = new double[inSize];
		output = new double[outSize];
		label = 0;
		//input_data = (double*) malloc(sizeof(double)*inSize);
		//output = (double*) malloc(sizeof(double)*outSize);
		//label = 0;
	}

};

#endif