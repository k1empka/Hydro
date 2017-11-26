/*
 * utils.cpp
 *
 *  Created on: 13 wrz 2017
 *      Author: mknap
 */
#include "utils.h"
#include "Fraction.h"
#include <helper_cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cmath>

FluidParams initParams()
{
    FluidParams params;
    params.d = make_float4(0.1, 0.1, 0.1, 0.04);
    params.omega = 0;
    params.mustaSteps = 4;
    return params;
}

void initCuda()
{
	int nDevices;
    int devCount = cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        cudaCheckErrors("Init Cuda")
        printf("CUDA device [%s] has %d Multi-Processors\n",
               props.name, props.multiProcessorCount);
    }
    if(nDevices > 1)
        cudaSetDevice(1); //Dla mnie bo mam SLI;
}

Fraction* initSpace(const bool random)
{
    Fraction* space = new Fraction[SIZE];
	if(NULL==space)
	{
		printf("memory allocation error\n");
		return NULL;
	}

	//IF RANDOM FLAG IS SET THEN INIT SPACE HAS DIFFERENT RESULT EACH TIME
	if(true==random)
		srand(time(NULL));


    const int X_MID = X_SIZE / 2;
    const int Y_MID = Y_SIZE / 2;
    const int Z_MID = Z_SIZE / 2;

	const float SPACE_FACTOR = .2;
	const int Z_SPACE = (int)ceil(Z_SIZE*SPACE_FACTOR);
	const int X_SPACE = (int)ceil(X_SIZE*SPACE_FACTOR);
	const int Y_SPACE = (int)ceil(Y_SIZE*SPACE_FACTOR);

	const float PLACE_FACTOR = 0.4;
	const int Z_PLACE = (int)ceil(Z_SIZE*PLACE_FACTOR);
	const int X_PLACE = (int)ceil(X_SIZE*PLACE_FACTOR);
	const int Y_PLACE = (int)ceil(Y_SIZE*PLACE_FACTOR);

	for(int z=0;z<Z_SPACE;++z)
	{
		for(int x=0; x<X_SPACE; ++x)
		{
			for(int y=0; y<Y_SPACE; ++y)
			{
				int idx = IDX_3D(X_PLACE+x,Y_PLACE+y,Z_PLACE+z);

				//IF RANDOM FLAG IS SET THEN INIT SPACE HAS DIFFRENT RESULT EACH TIME
                space[idx] = (random) ? Fraction(rand() % MAX_START_FORCE + 1, rand() % MAX_START_FORCE + 1, make_float3(MAX_VELOCITY%rand(), 0, 0)) :
                                        Fraction(MAX_START_FORCE, MAX_START_FORCE, make_float3(MAX_VELOCITY, 0, 0));
			}
		}
	}

	return space;
}

void swapPointers(void*& p1,void*& p2)
{
	void* tmp;

	tmp=p1;
	p1=p2;
	p2=tmp;
}

void compare_results(Fraction* hostSpace,Fraction* deviceSpace)
{
	float diffMax=0,diffMin=0;
	int numOfDiffs=0;
	bool firstDiff = true;

	for(int i=0; i<SIZE; ++i)
	{
        auto const& h = hostSpace[i];
        auto const& d = deviceSpace[i];
		if(hostSpace[i].E != deviceSpace[i].E)
		{
            numOfDiffs++;
		}	
	}

	printf("Compare results:\n\tNum of differences: %d\n",numOfDiffs);
}

void printData(float* data)
{
	printf("Data:\n");

	for(int y = 0;y < Y_SIZE;++y)
	{
		for(int x = 0;x < X_SIZE;++x)
		{
			printf("%1f  ",data[IDX_3D(x,y,0)]);
		}
		printf("\n");
	}
	printf("\n");
}

float* spaceToFloats(Fraction* space)
{
    float* spaceFloats = (float*)(malloc(sizeof(float)*SIZE*5));
	if(NULL==spaceFloats)
	{
		printf("memory allocation error\n");
		return NULL;
	}

	for(int i=0; i<SIZE;++i)
	{
		spaceFloats[5*i] = space[i].E;
		spaceFloats[5*i+1]=space[i].R;
		spaceFloats[5*i+2]=space[i].Vx;
		spaceFloats[5*i+3]=space[i].Vy;
		spaceFloats[5*i+4]=space[i].Vz;
	}

	return spaceFloats;
}

void floatsToSpace(float* floats,Fraction* space)
{
	for(int i=0; i<SIZE;++i)
	{
		space[i].E = floats[5*i];
		space[i].R = floats[5*i+1];
		space[i].Vx= floats[5*i+2];
		space[i].Vy= floats[5*i+3];
		space[i].Vz= floats[5*i+4];
	}
}



