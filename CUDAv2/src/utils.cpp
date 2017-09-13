/*
 * utils.cpp
 *
 *  Created on: 13 wrz 2017
 *      Author: mknap
 */
#include "utils.h"
#include <helper_cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cmath>

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

fraction* initSpace()
{
	fraction* space = (fraction*)malloc(sizeof(fraction));

	if(NULL==space)
	{
		printf("memory allocation error\n");
		return NULL;
	}

	for(int z=0; z<Z_SIZE; ++z)
	{
		for(int y=0; y<Y_SIZE; ++y)
		{
			for(int x=0; x<X_SIZE;++x)
			{
				space->U[IDX_3D(x,y,z)]=0.;
			}
		}
	}

	//srand(time(NULL));

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
				space->U[idx]= (float)(rand()%MAX_START_FORCE + 1);
				// We don't use it for now
				//space->Vx[X_PLACE+x]= (float)(rand()%MAX_START_FORCE + 1 - MAX_START_FORCE/2) * 0.05;
				//space->Vy[Y_PLACE+y]= (float)(rand()%MAX_START_FORCE + 1 - MAX_START_FORCE/2) * 0.01;
			}
		}
	}

	return space;
}

FILE* initOutputFile(bool hostSimulation)
{
	char filename[100];
	if(hostSimulation)
		sprintf(filename,"hostResult");
	else
		sprintf(filename,"result");
	FILE *f = fopen(filename, "wb");
	if (f == NULL)
	{
	    printf("Error opening file!\n");
	    exit(1);
	}
	return f;
}

void printHeader(FILE* f)
{
	int16_t x=(int16_t)X_SIZE;
	int16_t y=(int16_t)Y_SIZE;
	int16_t z=(int16_t)Z_SIZE;
	int16_t i=(int16_t)NUM_OF_ITERATIONS;
	int16_t floatSize=(int16_t)sizeof(float);
	int size=sizeof(int16_t);
	fwrite(&x,size,1,f);
	fwrite(&y,size,1,f);
	fwrite(&z,size,1,f);
	fwrite(&i,size,1,f);
	fwrite(&floatSize,size,1,f);
}

void printIteration(FILE* f,fraction* space, int iter)
{
	float v;
	int size = sizeof(float);


	for(int z=0; z<Z_SIZE;++z)
	{
		for(int y=0; y<Y_SIZE;++y)
		{
			for(int x=0; x<X_SIZE;++x)
			{
				v =space->U[IDX_3D(x,y,z)];
				fwrite(&v,size,1,f);
			}
		}
	}
}

void swapFractionPointers(fraction*& p1,fraction*& p2)
{
	fraction* tmp;

	tmp=p1;
	p1=p2;
	p2=tmp;
}




