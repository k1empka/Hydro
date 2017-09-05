#include <helper_cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "computation.cuh"
#include "Timer.h"

void initCuda()
{
	int nDevices;
    int devCount = cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp props;
        checkCudaErrors(cudaGetDeviceProperties(&props, i));
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

	srand (time(NULL));

	//W TYM FORZE JEST JAKIS BLAD ALE NIE WIEM JAKI...

	for(int z=0;z<20;++z)
	{
		for(int x=0; x<20; ++x)
		{
			for(int y=0; y<20; ++y)
			{
				int idx = IDX_3D(40+x,40+y,40+z);
				space->U[idx]= (float)(rand()%MAX_START_FORCE + 1);
				space->Vx[idx]= (float)(rand()%MAX_START_FORCE + 1 - MAX_START_FORCE/2) * 0.05;
				space->Vy[idx]= (float)(rand()%MAX_START_FORCE + 1 - MAX_START_FORCE/2) * 0.01;
			}
		}
	}

	//POWYZEJ JEST JAKIS BLAD ALE NIE WIEM JAKI...

	return space;
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

inline void swapFractionPointers(fraction*& p1,fraction*& p2)
{
	fraction* tmp;

	tmp=p1;
	p1=p2;
	p2=tmp;
}

int main()
{
	bool hostSimulationOn = false;

	initCuda();

	fraction* space = initSpace();

	if(NULL==space)
		return -1;

	fraction *d_space,*d_result;
	int totalSize=sizeof(fraction);
	cudaMalloc((void **)&d_space,totalSize);
	cudaMalloc((void **)&d_result,totalSize);
	cudaMemcpy(d_space,space,totalSize, cudaMemcpyHostToDevice);

	FILE* f = initOutputFile(false);

	printHeader(f);
	printf("Simulation started\n");
	for(int i=0;i<NUM_OF_ITERATIONS;++i)
	{
		Timer::getInstance().start("Device simulation time");

		if((i % 2) != 0)
		{
			swapFractionPointers(d_space,d_result);
		}

		simulation(d_space,d_result);
		cudaMemcpy(space,d_result,totalSize, cudaMemcpyDeviceToHost);
		Timer::getInstance().stop("Device simulation time");
		printIteration(f,space,i);
	}
	printf("Simulation completed\n");
	Timer::getInstance().printResults();

	cudaFree(d_space);
	cudaFree(d_result);
	free(space);
	fclose(f);

	if(hostSimulationOn)
	{
		Timer::getInstance().clear();
		fraction* result=(fraction*)malloc(totalSize);
		if(NULL == result)
			return -1;

		space=initSpace();//PRZY DRUGIM WYWOLANIU TEJ FUNKCJI JEST CORE DUMP

		f = initOutputFile(hostSimulationOn);

		printHeader(f);

		printf("Host simulation started\n");
		for(int i=0;i<NUM_OF_ITERATIONS;++i)
		{
			Timer::getInstance().start("Host simulation time");

			if((i % 2) != 0)
			{
				swapFractionPointers(space,result);
			}

			hostSimulation(space,result);
			Timer::getInstance().stop("Host simulation time");
			printIteration(f,space,i);
		}
		printf("Host simulation completed\n");
		Timer::getInstance().printResults();

		cudaFree(d_space);
		cudaFree(d_result);
		free(space);
		fclose(f);
	}

	return 0;
}
