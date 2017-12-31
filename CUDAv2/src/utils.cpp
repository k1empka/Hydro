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
	if(nullptr==space)
	{
		printf("memory allocation error\n");
		return nullptr;
	}

	//IF RANDOM FLAG IS SET THEN INIT SPACE HAS DIFFERENT RESULT EACH TIME
	if(true==random)
		srand(time(NULL));

    const int3 mid = make_int3(X_SIZE / 2, Y_SIZE / 2, Z_SIZE / 2);
    const int3 rad = make_int3(X_SIZE / 6, Y_SIZE / 6, Z_SIZE / 6);
    const int3 start = make_int3(mid.x - rad.x, mid.y - rad.y, mid.z - rad.z);
    const int3 end = make_int3(mid.x + rad.x, mid.y + rad.y, mid.z + rad.z);

	for(int z=start.z; z < end.z;++z)
        for (int y = start.y; y < end.y; ++y)
            for (int x = start.x; x < end.x; ++x)
	        {
                int i = IDX_3D(x, y, z);
		        if(true==random)
		        {
			        space[i].E = (float)(rand() % MAX_START_FORCE + 1);
			        space[i].R = (float)(rand() % MAX_START_FORCE + 1);
			        space[i].Vx= (float)(MAX_VELOCITY%rand());
			        space[i].Vy= 0.0f;
			        space[i].Vz= 0.0f;
		        }
		        else
		        {
			        space[i].E = (float)MAX_START_FORCE;
			        space[i].R = (float)MAX_START_FLUX;
			        space[i].Vx= (float)MAX_VELOCITY;
			        space[i].Vy= 0.0f;
			        space[i].Vz= 0.0f;
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
    float eps = 0.001;

    for (int z = 2; z<Z_SIZE - 2; ++z)
    {
        for (int y = 2; y < Y_SIZE - 2; ++y)
        {
            for (int x = 2; x < X_SIZE - 2; ++x)
            {
                int i = IDX_3D(x, y, z);
                auto const& h = hostSpace[i];
                auto const& d = deviceSpace[i];
                float err = fabs(hostSpace[i].E - deviceSpace[i].E);
                if (err > eps)
                {
                    numOfDiffs++;
                }
            }
        }
	}

	printf("Compare results:\n\tNum of differences: %d\n",numOfDiffs);
}

void printData(Fraction* space)
{
	printf("Data:\n");

	for(int i=0; i<SIZE;++i)
	{
		printf("id:%d\tE:%f\tR:%f\tVx:%f\tVy:%f\tVz:%f\n",i,space[i].E,space[i].R,space[i].Vx,space[i].Vy,space[i].Vz);
	}
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



