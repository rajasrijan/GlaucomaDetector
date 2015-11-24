#include "SelfOrganisingMap.h"
#include <stdarg.h>
#include <stdint.h>
#include <time.h>


SelfOrganisingMap::SelfOrganisingMap(int inputs,int dimensions,...)	:	mInputs(inputs),mDimentions(dimensions)
{
	va_list ap;
	va_start(ap,dimensions);
	for (int i = 0; i < mDimentions; i++)
	{
		int sz=va_arg(ap,int);
		szDimentions.push_back(sz);
	}
	va_end(ap);
	create_everything();
}


SelfOrganisingMap::~SelfOrganisingMap(void)
{
	destroy_everything();
}

void SelfOrganisingMap::destroy_everything()
{
	mInputs=0;
	delete weight;
	weight=0;
	delete data;
	data=0;
	mDimentions=0;
	mTotalNeurons=0;
	szDimentions.clear();
	sigma=0;
	divident=0;
	learn=0;
}

void SelfOrganisingMap::create_everything()
{
	int no_neurons=1;
	divident=exp(-1/1000000.0f);
	learn=0.9;
	sigma=0;
	for (int i = 0; i < szDimentions.size(); i++)
	{
		no_neurons*=szDimentions[i];
		if(szDimentions[i]>sigma)
			sigma=szDimentions[i];
	}
	sigma/=2;
	mTotalNeurons=no_neurons;
	size_t data_size=no_neurons;
	size_t weight_size=mInputs*no_neurons;
	data=new double[data_size];
	weight=new double[weight_size];
	uint32_t seed=time(0);
	for (int i = 0; i < weight_size; i++)
	{
		seed*=seed;
		weight[i]=(double)((seed+1)%256);
	}
}

void SelfOrganisingMap::train(double* input_data)
{
	int index=0;
	double min_distance=99999999.9f;
	for (int i = 0; i < mTotalNeurons; i++)
	{
		double dist=0;
		for (int j = 0; j < mInputs; j++)
		{
			dist += pow((weight[(i*mInputs) + j] - input_data[j]),2);
		}
		dist=sqrt(dist);
		if (dist<min_distance)
		{
			index=i;
			min_distance=dist;
		}
	}

	int tmp=index;
	std::vector<int> BMUposition;
	for (int j = 0; j  < mDimentions; j ++)
	{
		BMUposition.push_back(tmp%szDimentions[j]);
		tmp/=szDimentions[j];
	}

	for (int i = 0; i < mTotalNeurons; i++)
	{
		/*
		*	Convert i to n-D vector and calculate distance.
		*/
		int tmp=i;
		std::vector<int> position;
		double dist=0;
		for (int j = 0; j  < mDimentions; j ++)
		{
			int cord = tmp%szDimentions[j];
			position.push_back(cord);
			tmp /= szDimentions[j];
			dist+=pow(BMUposition[j]-cord,2);
		}
		dist=sqrt(dist);

		/*
		*	Check if in range.
		*/
		//if (dist<=sigma)
		//{
		//	Calculate distance factor.
		double rad=exp(-pow(dist/sigma,2)/2.0f);
		//	Update weights.
		for (int j = 0; j < mInputs; j++)
			weight[(i*3) + j] = weight[(i*3) + j] + (rad*learn*(input_data[j] - weight[(i*3) + j]));
		//}
	}
	sigma*=divident;;
	learn*=divident;
}

void SelfOrganisingMap::calculate(double* input_data)
{
	for (int i = 0; i < mTotalNeurons; i++)
	{
		double dist=0;
		for (int j = 0; j < mInputs; j++)
		{
			dist += pow((weight[(i*mInputs) + j] - input_data[j]),2);
		}
		data[i]=sqrt(dist);
	}
}

void SelfOrganisingMap::writeOutput(FILE *fp,char seperator)
{
	for (uint32_t i = 0; i < mTotalNeurons; i++)
		fprintf(fp,"%f%c",data[i],seperator);
	fprintf(fp,"\n");
}

bool SelfOrganisingMap::serialize(FILE* fp)
{
	if (!fp)
	{
		return 1;
	}

	size_t weight_size=mTotalNeurons*mInputs;


	int ret=0;
	ret=fprintf(fp,"SOM\n");					//	Signature.
	if(ret<=0)
		return 1;
	ret=fprintf(fp,"DIMENTIONS=%d\n",mDimentions);		//	Number of dimentions
	if(ret<=0)
		return 1;
	ret=fprintf(fp,"SZDIMENTIONS\n");	//	Number of neurons per dimention.
	if(ret<=0)
		return 1;
	for (int i = 0; i < mDimentions; i++)
	{
		ret=fprintf(fp,"%d,",szDimentions[i]);	//	Number of neurons.
		if(ret<=0)
			return 1;
	}
	fprintf(fp,"\n");

	ret=fprintf(fp,"INPUTS=%d\n",mInputs);		//	Number of inputs.
	if(ret<=0)
		return 1;

	ret=fprintf(fp,"WEIGHTS\n");				//	Comma seperated weights
	if(ret<=0)
		return 1;

	for (uint32_t i = 0; i < weight_size; i++)
	{
		ret=fprintf(fp,"%f,",weight[i]);
		if(ret<=0)
			return 1;
	}

	fflush(fp);
	return 0;
}

bool SelfOrganisingMap::deserialize(FILE *fp)
{	
	if (!fp)
	{
		return 1;
	}

	destroy_everything();

	int ret=0;
	char signature[16]={0};
	ret=fscanf(fp,"%s\n",signature);		//	Signature.
	if(ret<=0)
		return 1;
	if (strcmp(signature,"SOM"))
		return 1;
	ret=fscanf(fp,"DIMENTIONS=%d\n",&mDimentions);		//	Number of layers
	if(ret<=0)
		return 1;
	ret=fscanf(fp,"%s\n",signature);	//	Number of neurons per layer.
	if(ret<=0)
		return 1;
	if (strcmp(signature,"SZDIMENTIONS"))
		return 1;
	mTotalNeurons=1;
	for (uint32_t i = 0; i < mDimentions; i++)
	{
		int val;
		ret=fscanf(fp,"%d,",&val);
		if(ret<=0)
			return 1;
		szDimentions.push_back(val);
		mTotalNeurons*=val;
	}
	fscanf(fp,"\n");
	ret=fscanf(fp,"INPUTS=%d\n",&mInputs);		//	Number of inputs.
	if(ret<=0)
		return 1;
	ret=fscanf(fp,"%s\n",signature);				//	Comma seperated weights
	if(ret<=0)
		return 1;
	if (strcmp(signature,"WEIGHTS"))
		return 1;


	create_everything();

	size_t weight_size=mTotalNeurons*mInputs;

	for (uint32_t i = 0; i < weight_size; i++)
	{
		float val;
		ret=fscanf(fp,"%f,",&val);
		weight[i]=val;
		if(ret<=0)
			return 1;
	}

	return 0;
}

int SelfOrganisingMap::getLowest()
{
	int index=0;
	for (int i = 1; i < mTotalNeurons; i++)
	{
		if (data[i]<data[index])
		{
			index=i;
		}
	}
	return index;
}