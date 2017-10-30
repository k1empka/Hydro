#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "customTimer.h"
#include "printer.h"

#define RANDOM false
#define PRINT_RESULTS false

fraction* execHost()
{
	Timer::getInstance().clear();
	int totalSize=sizeof(fraction);
	void* space,*result=(fraction*)malloc(totalSize);
    if (NULL == result)
    {
        printf("Malloc problem!\n");
        exit(-1);
    }

	space = initSpace(RANDOM);
#if PRINT_RESULTS
    Printer bytePrinter("host.data");
#endif
	printf("Host simulation started\n");
    Timer::getInstance().start("Host simulation time");
	for(int i=0;i<NUM_OF_ITERATIONS;++i)
	{
		if((i % 2) != 0)
		{
			swapPointers(space,result);
		}
		hostSimulation(space,result);
#if PRINT_RESULTS
        bytePrinter.printIteration(result, i);
#endif
	}
    Timer::getInstance().stop("Host simulation time");
	printf("Host simulation completed\n");
	Timer::getInstance().printResults();

	free(space);
	return (fraction*)result;
}

fraction* execDevice(enum deviceSimulationType type)
{
	fraction* space = initSpace(RANDOM);

	if(NULL==space)
		exit(-1);

	void *d_space,*d_result;
	int totalSize = sizeof(fraction);
	cudaMalloc((void **)&d_space,totalSize);
	cudaMalloc((void **)&d_result,totalSize);
	cudaMemcpy(d_space,space,totalSize, cudaMemcpyHostToDevice);

#if PRINT_RESULTS
    Printer bytePrinter("device.data");
#endif
	printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

	for(int i=0;i<NUM_OF_ITERATIONS;++i)
	{
		if((i % 2) != 0)
		{
			swapPointers(d_space,d_result);
		}
		simulation(d_space,d_result,type);
#if PRINT_RESULTS
        cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);
        bytePrinter.printIteration(space, i);
#endif
	}
#if !PRINT_RESULTS
    cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);
#endif
    Timer::getInstance().stop("Device simulation time");
	printf("Simulation completed\n");
	Timer::getInstance().printResults();

	cudaFree(d_space);
	cudaFree(d_result);
	return space;
}
int main()
{
	bool hostSimulationOn = true;
	enum deviceSimulationType type = GLOBAL;

	fraction* hostOutputSpace,* deviceOutputSpace;

	initCuda();
	deviceOutputSpace = execDevice(type);

	if(hostSimulationOn)
	{
		hostOutputSpace = execHost();
	}
    compare_results(hostOutputSpace,deviceOutputSpace);
	free(hostOutputSpace);
	free(deviceOutputSpace);
	return 0;
}
