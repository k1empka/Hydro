#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "customTimer.h"
#include "printer.h"
#include "Fraction.h"

#define RANDOM false
#define PRINT_RESULTS false

Fraction* execHost()
{
	Timer::getInstance().clear();
	int totalSize=sizeof(Fraction) * SIZE;
	void* space,*result=(Fraction*)malloc(totalSize);
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
		hostSimulation(initParams(),space,result);
#if PRINT_RESULTS
        bytePrinter.printIteration(result, i);
#endif
	}
    Timer::getInstance().stop("Host simulation time");
	printf("Host simulation completed\n");
	Timer::getInstance().printResults();

	free(space);
	return (Fraction*)result;
}

Fraction* execDeviceSurface(Fraction* space)
{
	int memSize=sizeof(float)*SIZE;

	// For float we could create a channel with:
	cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);

	// Allocate memory in device
	cudaArray* cuSpaceArray;
	cudaMallocArray(&cuSpaceArray, &channelDesc, SIZE,cudaArraySurfaceLoadStore);
	cudaArray* cuResultArray;
	cudaMallocArray(&cuResultArray, &channelDesc, SIZE,cudaArraySurfaceLoadStore);

	// Copy to device memory initial data
	cudaMemcpyToArray(cuSpaceArray, 0, 0, space, memSize,cudaMemcpyHostToDevice);

	// Specify surface
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	// Create the surface objects
	resDesc.res.array.array = cuSpaceArray;
	cudaSurfaceObject_t spaceSurfObj=0;
	cudaCreateSurfaceObject(&spaceSurfObj, &resDesc);
	resDesc.res.array.array = cuResultArray;
	cudaSurfaceObject_t resultSurfObj=0;
	cudaCreateSurfaceObject(&resultSurfObj, &resDesc);

#if PRINT_RESULTS
    Printer bytePrinter("device.data");
#endif
	printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

	for(int i=0;i<NUM_OF_ITERATIONS;++i)
	{
		if(i%2!=0)
		{
			simulationSurface(resultSurfObj,spaceSurfObj);
#if PRINT_RESULTS
		cudaMemcpyFromArray(space,cuSpaceArray, 0, 0, memSize,cudaMemcpyDeviceToHost);
        bytePrinter.printIteration(space, i);
#endif
		}
		else
		{
			simulationSurface(spaceSurfObj,resultSurfObj);
#if PRINT_RESULTS
		cudaMemcpyFromArray(space,cuResultArray, 0, 0, memSize,cudaMemcpyDeviceToHost);
        bytePrinter.printIteration(space, i);
#endif
		}
	}
#if !PRINT_RESULTS
	cudaMemcpyFromArray(space,cuResultArray, 0, 0, memSize,cudaMemcpyDeviceToHost);
#endif
    Timer::getInstance().stop("Device simulation time");
	printf("Simulation completed\n");
	Timer::getInstance().printResults();

	// Destroy surface objects
	cudaDestroySurfaceObject(spaceSurfObj);
	cudaDestroySurfaceObject(resultSurfObj);

	// Free device memory
	cudaFreeArray(cuSpaceArray);
	cudaFreeArray(cuResultArray);

	return space;
}

Fraction* execDevice(enum deviceSimulationType type)
{
	Fraction* space = initSpace(RANDOM);

	if(NULL==space)
		exit(-1);

	//DUE TO PROBLEMS WITH POINTERS AND SURFACE MEMORY OBJECTS THIS KIND OF SIMULATION IS THREATED SEPARATELY
	if(SURFACE==type)
		return execDeviceSurface(space);
    FluidParams params = initParams();
	void *d_space,*d_result,*d_params;
	int totalSize = sizeof(Fraction)*SIZE;
	cudaMalloc((void **)&d_space,totalSize);
	cudaMalloc((void **)&d_result,totalSize);
    cudaMalloc((void **)&d_params, sizeof(FluidParams));
	cudaMemcpy(d_space,space,totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &params, sizeof(FluidParams), cudaMemcpyHostToDevice);

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
		simulation(d_params,d_space,d_result,type);
#if PRINT_RESULTS
        cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);
        bytePrinter.printIteration(space, i);
#else
        cudaThreadSynchronize();
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

	Fraction* hostOutputSpace,* deviceOutputSpace;

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
