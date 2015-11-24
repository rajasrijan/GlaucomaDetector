#include "FeedForward.h"


FeedForward::FeedForward(uint32_t inputs,uint32_t outputs,uint32_t layers,uint32_t neurons)	:	mInputs(inputs),mOutputs(outputs),mLayers(layers),mNeurons(neurons)
{
	create_everything();
}

FeedForward::~FeedForward(void)
{
	destroy_everything();
}

void FeedForward::calculate(double* input_data)
{
	/*
	* First layer.
	*/
	for (uint32_t i = 0; i < mNeurons; i++)
	{
		double tmp=0;
		for (uint32_t j = 0; j < mInputs; j++)
		{
			tmp+=(input_data[j]*weight[(j*mNeurons)+i]);
		}
		data[0+i]=Fn(tmp);
	}

	/*
	*	Hidden layers.
	*/
	for (uint32_t i = 1; i < mLayers; i++)
	{
		for (uint32_t j = 0; j < mNeurons; j++)
		{
			double tmp=0;
			for (uint32_t k = 0; k < mNeurons; k++)
			{
				tmp+=(data[(mNeurons*(i-1))+k]*weight[(mInputs*mNeurons) + (mNeurons*mNeurons*(i-1)) + (mNeurons*k) + j]);
			}
			data[(mNeurons*i)+j]=Fn(tmp);
		}
	}

	/*
	*	Output layer.
	*/
	for (uint32_t i = 0; i < mOutputs; i++)
	{
		double tmp=0;
		for (uint32_t j = 0; j < mNeurons; j++)
		{
			tmp+=(weight[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (j*mOutputs) + i]*data[((mLayers-1)*mNeurons) + j]);
		}
		data[(mLayers*mNeurons) + i]=Fn(tmp);
	}
}

void FeedForward::writeOutput(FILE *fp,char seperator)
{
	for (uint32_t i = 0; i < mOutputs; i++)
		fprintf(fp,"%f%c",data[(mLayers*mNeurons) + i],seperator);
	fprintf(fp,"\n");
}

bool FeedForward::serialize(FILE* fp)
{
	if (!fp)
	{
		return 1;
	}

	size_t weight_size=(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (mOutputs*mNeurons);


	int ret=0;
	ret=fprintf(fp,"FFNN\n");					//	Signature.
	if(ret<=0)
		return 1;
	ret=fprintf(fp,"LAYERS=%d\n",mLayers);		//	Number of layers
	if(ret<=0)
		return 1;
	ret=fprintf(fp,"NEURONS=%d\n",mNeurons);	//	Number of neurons per layer.
	if(ret<=0)
		return 1;
	ret=fprintf(fp,"INPUTS=%d\n",mInputs);		//	Number of inputs.
	if(ret<=0)
		return 1;
	ret=fprintf(fp,"OUTPUTS=%d\n",mOutputs);	//	Number of outputs.
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

bool FeedForward::deserialize(FILE *fp)
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
	if (strcmp(signature,"FFNN"))
		return 1;
	ret=fscanf(fp,"LAYERS=%d\n",&mLayers);		//	Number of layers
	if(ret<=0)
		return 1;
	ret=fscanf(fp,"NEURONS=%d\n",&mNeurons);	//	Number of neurons per layer.
	if(ret<=0)
		return 1;
	ret=fscanf(fp,"INPUTS=%d\n",&mInputs);		//	Number of inputs.
	if(ret<=0)
		return 1;
	ret=fscanf(fp,"OUTPUTS=%d\n",&mOutputs);	//	Number of outputs.
	if(ret<=0)
		return 1;
	ret=fscanf(fp,"%s\n",signature);				//	Comma seperated weights
	if(ret<=0)
		return 1;
	if (strcmp(signature,"WEIGHTS"))
		return 1;


	create_everything();

	size_t weight_size=(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (mOutputs*mNeurons);

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

void FeedForward::destroy_everything()
{
	mLayers=0;
	mNeurons=0;
	mInputs=0;
	mOutputs=0;
	delete weight;
	weight=0;
	delete delta;
	delta=0;
	delete dE_by_dW_tminus;
	dE_by_dW_tminus=0;
	delete data;
	data=0;
	delete error;
	error=0;
}

void FeedForward::create_everything()
{
	size_t data_size=(mLayers*mNeurons) + mOutputs;
	size_t weight_size=(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (mOutputs*mNeurons);
	data=new double[data_size];
	weight=new double[weight_size];
	delta=new double[weight_size];
	dE_by_dW_tminus=new double[weight_size];
	for (uint32_t i = 0; i < weight_size; i++)
	{
		delta[i]=0.1;
		dE_by_dW_tminus[i]=1.0f;
		weight[i]=1.0f;
	}
	error=new double[data_size];
	for (uint32_t i = 0; i < data_size; i++)
		error[i]=(double)(i+1);
}

void FeedForward::rprop()
{
	/*
	*	RPROP last hidden layer.
	*/
	for (uint32_t i = 0; i < mNeurons; i++)
	{
		double	dE=0;
		for (uint32_t j = 0; j < mOutputs; j++)
		{
			/*
			*	Propogate error.
			*/
			dE+=(error[(mLayers*mNeurons) + j]*weight[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j]);

			/*
			*	Perform training.
			*/
			double E=error[(mLayers*mNeurons) + j];
			double derivative=1;
			double dE_by_dW=E*derivative*data[((mLayers-1)*mNeurons) + i];

			double d=0;
			double dW = 0;
			double s=dE_by_dW_tminus[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j]*dE_by_dW;

			if (s>0)
				d=1.2*delta[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j];
			else if (s<0)
				d=0.5*delta[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j];
			else
				d=delta[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j];

			if (s>=0)
			{
				if (dE_by_dW>0)
					dW=d;
				else if (dE_by_dW<0)
					dW=-d;
				else
					dW=0;
			}
			else
			{
				/*
				*	Calculate previous dW.
				*/
				if (dE_by_dW_tminus[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j]>0)
					dW=delta[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j];
				else if (dE_by_dW_tminus[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j]<0)
					dW=-delta[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j];
				else
					dW=0;
				//dE_by_dW=0;
				/*
				*	dW(t) = -dW(t-1)
				*/
				dW=-dW;
			}


			/*
			*	Save results for next iteration.
			*/
			delta[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j]=d;
			dE_by_dW_tminus[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j]=dE_by_dW;
			/*
			*	Update weight.
			*/
			weight[(mNeurons*(mLayers-1)*mNeurons) + (mInputs*mNeurons) + (i*mOutputs) + j]-=dW;
		}

		error[((mLayers-1)*mNeurons) + i]=dE;
	}

	/*
	*	RPROP remaining hidden layers.
	*/
	for (int i=mLayers-2; i >=0; i--)
	{
		for (uint32_t j = 0; j < mNeurons; j++)
		{
			double dE=0;
			for (uint32_t k = 0; k < mNeurons; k++)
			{
				/*
				*	Propogate error.
				*/
				dE+=(error[((i + 1)*mNeurons) + k]*weight[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k]);
				/*
				*	Perform training.
				*/
				//	Error from next layer.
				double E=error[((i+1)*mNeurons) + k];
				//	Derivative of the sigma function.
				double derivative=1;
				//	dE/dW_ij of current weight.
				double dE_by_dW=E*derivative*data[((i)*mNeurons) + j];

				double d=0;
				double dW = 0;
				double s=dE_by_dW_tminus[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k]*dE_by_dW;

				if (s>0)
					d=1.2*delta[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k];
				else if (s<0)
					d=0.5*delta[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k];
				else
					d=delta[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k];

				if (s>=0)
				{
					if (dE_by_dW>0)
						dW=d;
					else if (dE_by_dW<0)
						dW=-d;
					else
						dW=0;
				}
				else
				{
					/*
					*	Calculate previous dW.
					*/
					if (dE_by_dW_tminus[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k]>0)
						dW=delta[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k];
					else if (dE_by_dW_tminus[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k]<0)
						dW=-delta[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k];
					else
						dW=0;

					//dE_by_dW=0;
					/*
					*	dW(t) = -dW(t-1)
					*/
					dW=-dW;
				}

				/*
				*	Save results for next iteration.
				*/
				delta[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k]=d;
				dE_by_dW_tminus[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k]=dE_by_dW;
				/*
				*	Update weight.
				*/
				weight[(mInputs*mNeurons) + (mNeurons*mNeurons*(i)) + (mNeurons*j) + k]-=dW;
			}
			error[(i*mNeurons)+j]=dE;
		}
	}

	/*
	*	RPROP first layer.
	*/
	for (uint32_t i = 0; i < mInputs; i++)
	{
		for (uint32_t j = 0; j < mNeurons; j++)
		{
			/*
			*	Input layer dosen't need error propogation.
			*	Input layer is only buffer.
			*/

			/*
			*	Perform training.
			*/
			//	Error from next layer.
			double E=error[0 + j];
			//	Derivative of the sigma function.
			double derivative=1;
			//	dE/dW_ij of current weight.
			double dE_by_dW=E*derivative*data[j];

			double d=0;
			double dW = 0;
			double s=dE_by_dW_tminus[(i*mNeurons) + j]*dE_by_dW;

			if (s>0)
				d=1.2*delta[(i*mNeurons) + j];
			else if (s<0)
				d=0.5*delta[(i*mNeurons) + j];
			else
				d=delta[(i*mNeurons) + j];

			if (s>=0)
			{
				if (dE_by_dW>0)
					dW=d;
				else if (dE_by_dW<0)
					dW=-d;
				else
					dW=0;
			}
			else
			{
				/*
				*	Calculate previous dW.
				*/
				if (dE_by_dW_tminus[(i*mNeurons) + j]>0)
					dW=delta[(i*mNeurons) + j];
				else if (dE_by_dW_tminus[(i*mNeurons) + j]<0)
					dW=-delta[(i*mNeurons) + j];
				else
					dW=0;

				//dE_by_dW=0;
				/*
				*	dW(t) = -dW(t-1)
				*/
				dW=-dW;
			}


			/*
			*	Save results for next iteration.
			*/
			delta[(i*mNeurons) + j]=d;
			dE_by_dW_tminus[(i*mNeurons) + j]=dE_by_dW;
			/*
			*	Update weight.
			*/
			weight[(i*mNeurons) + j]-=dW;
		}
	}
}

double FeedForward::Fn(double in)
{
	return in;
}

double FeedForward::dFn(double in)
{
	return 1;
}

void FeedForward::acumulateError(double* output_data)
{
	/*
	*	RPROP output layer.
	*/
	for (uint32_t i = 0; i < mOutputs; i++)
	{
		double	dE=0;
		dE = ( data[(mLayers*mNeurons) + i] - output_data[i] );
		error[(mLayers*mNeurons) + i]+=dE;
	}
}

void FeedForward::clearError()
{
	for (uint32_t i = 0; i < mOutputs; i++)
	{
		error[(mLayers*mNeurons) + i]=0;
	}
}
