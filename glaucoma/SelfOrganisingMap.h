#pragma once
#include <vector>
class SelfOrganisingMap
{
private:
	int mInputs;
	int mDimentions;
	int mTotalNeurons;
	std::vector<int> szDimentions;
	double *data;
	double *weight;
	double sigma;
	double divident;
public:
	double learn;
private:
	void destroy_everything();
	void create_everything();
public:
	SelfOrganisingMap(int inputs,int dimensions,...);
	~SelfOrganisingMap(void);
	void train(double* input_data);
	void calculate(double* input_data);
	void writeOutput(FILE *fp=stdout,char seperator=' ');
	bool serialize(FILE* fp=stdout);
	bool deserialize(FILE* fp);
	int getLowest();
};

