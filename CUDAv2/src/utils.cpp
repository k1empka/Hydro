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
    params.d = make_float4(1, 1, 1, 1);
    params.omega = 1;
    params.mustaSteps = 10;
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

    space[IDX_3D(X_MID, Y_MID, Z_MID)] = Fraction(100,100,make_float3(5.,0.,0.));

/*	const float SPACE_FACTOR = .2;
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
				space[idx]= (float) ((true == random) ? (rand()%MAX_START_FORCE + 1) : (MAX_START_FORCE));

				// We don't use it for now
				//space->Vx[X_PLACE+x]= (float)(rand()%MAX_START_FORCE + 1 - MAX_START_FORCE/2) * 0.05;
				//space->Vy[Y_PLACE+y]= (float)(rand()%MAX_START_FORCE + 1 - MAX_START_FORCE/2) * 0.01;
			}
		}
	}*/

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



