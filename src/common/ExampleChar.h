#ifndef exampleChar_h
#define exampleChar_h

class ExampleChar
{
public:
	double * input_data;
	double * output;
	int label;

	ExampleChar(int inSize, int outSize)
	{
		input_data = (double*) malloc(sizeof(double)*inSize);
		output = (double*) malloc(sizeof(double)*outSize);
		label = 0;
	}

};

#endif