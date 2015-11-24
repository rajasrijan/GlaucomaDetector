#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#pragma once
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cmath>

class FeedForward
{
private:
	/*
	*	connecting weights of the newrons.
	*/
	double *weight;
	/*
	*	Deltas of the weights.
	*/
	double *delta;
	/*
	*	't-1' dE/dW of a weight.
	*/
	double *dE_by_dW_tminus;
	/*
	*	place where results of calculation are stored.
	*/
	double *data;
	/*
	*	place where error is stored.
	*/
	double *error;
	/*
	*	Number of layers.
	*/
	uint32_t mLayers;
	/*
	*	Number of neurons per layers.
	*/
	uint32_t mNeurons;
	/*
	*	Number of inputs.
	*/
	uint32_t mInputs;
	/*
	*	Number of outputs.
	*/
	uint32_t mOutputs;
public:
	FeedForward(uint32_t inputs,uint32_t outputs,uint32_t layers,uint32_t neurons);
	~FeedForward(void);
	void calculate(double* input_data);
	void writeOutput(FILE *fp=stdout,char seperator=' ');
	bool serialize(FILE* fp=stdout);
	bool deserialize(FILE* fp);
	void rprop();
	void acumulateError(double* output_format);
	void clearError();
private:
	/*
	*	Destroy and deallocate the complete network.
	*/
	void destroy_everything();
	/*
	*	Create and allocate the complete network.
	*/
	void create_everything();
	double Fn(double in);
	double dFn(double in);
};
