#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "customTimer.h"


void execHost()
{
	Timer::getInstance().clear();
	int totalSize=sizeof(fraction);
	fraction* space,*result=(fraction*)malloc(totalSize);
	FILE* f;

	if(NULL == result)
		exit(-1);

	space=initSpace();

	f = initOutputFile(true);

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

	free(space);
	free(result);
	fclose(f);
}

void execDevice()
{
	fraction* space = initSpace();

	if(NULL==space)
		exit(-1);

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
}
int main()
{
	bool hostSimulationOn = true;

	initCuda();
	execDevice();

	if(hostSimulationOn)
	{
		//execHost();
	}

	return 0;
}
