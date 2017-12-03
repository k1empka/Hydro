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
    int totalSize=sizeof(Fraction) * SIZE, i;
    void* space,*result= new Fraction[totalSize];
    auto params = initParams();
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
    for(i=0;i<NUM_OF_ITERATIONS;++i)
    {
    	hostSimulation(&params,space,result);
		swapPointers(space,result);
#if PRINT_RESULTS
		if(i % 2==0)
			bytePrinter.printIteration((Fraction*)result, i);
		else
			bytePrinter.printIteration((Fraction*)space, i);
#endif
    }
    Timer::getInstance().stop("Host simulation time");
    printf("Host simulation completed\n");
    Timer::getInstance().printResults();

    if(i % 2 == 0)
    {
		free(space);
		return (Fraction*)result;
    }
    else
    {
    	free(result);
		return (Fraction*)space;
    }
}

Fraction* execDeviceSurface(FluidParams* params,Fraction* space)
{
    const int memSize=sizeof(float)*SIZE*5;
    float* floats =spaceToFloats(space);

    // For float we could create a channel with:
    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc<float>();

    // Allocate memory in device
    cudaArray* cuSpaceArray;
    cudaMallocArray(&cuSpaceArray, &channelDesc, SIZE*5,cudaArraySurfaceLoadStore);
    cudaArray* cuResultArray;
    cudaMallocArray(&cuResultArray, &channelDesc, SIZE*5,cudaArraySurfaceLoadStore);

    // Copy to device memory initial data
    cudaMemcpyToArray(cuSpaceArray, 0, 0, floats, memSize,cudaMemcpyHostToDevice);

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

    void *resultObjPointer=&resultSurfObj,*spaceObjPointer=&spaceSurfObj;
    int i=0;

#if PRINT_RESULTS
    Printer bytePrinter("device.data");
#endif
    printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

    for(;i<NUM_OF_ITERATIONS;++i)
    {
    	simulationSurface(params,*(cudaSurfaceObject_t*)spaceObjPointer,*(cudaSurfaceObject_t*)resultObjPointer);
		swapPointers(spaceObjPointer,resultObjPointer);

#if PRINT_RESULTS
        if(i % 2 == 0)
            cudaMemcpyFromArray(floats,cuResultArray, 0, 0, memSize,cudaMemcpyDeviceToHost);
        else
            cudaMemcpyFromArray(floats,cuSpaceArray, 0, 0, memSize,cudaMemcpyDeviceToHost);
#endif
    }
#if !PRINT_RESULTS
    if(i % 2 == 0)
		cudaMemcpyFromArray(floats,cuResultArray, 0, 0, memSize,cudaMemcpyDeviceToHost);
	else
		cudaMemcpyFromArray(floats,cuSpaceArray, 0, 0, memSize,cudaMemcpyDeviceToHost);
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

    floatsToSpace(floats,space);

    //free temporary memory
    free(floats);

    return space;
}

Fraction* execDevice(enum deviceSimulationType type)
{
    Fraction* space = initSpace(RANDOM);
    FluidParams *d_params,params = initParams();
    cudaMalloc((void **)&d_params, sizeof(FluidParams));
    cudaMemcpy(d_params, &params, sizeof(FluidParams), cudaMemcpyHostToDevice);

    if(NULL==space)
        exit(-1);

    //DUE TO PROBLEMS WITH POINTERS AND SURFACE MEMORY OBJECTS THIS KIND OF SIMULATION IS THREATED SEPARATELY
    if(SURFACE==type)
        return execDeviceSurface(d_params,space);
    void *d_space,*d_result;
    int totalSize = sizeof(Fraction)*SIZE, i;
    void *result = new Fraction[totalSize];

    cudaMalloc((void **)&d_space,totalSize);
    cudaMalloc((void **)&d_result,totalSize);
    cudaCheckErrors("Mallocs");
    cudaMemcpy(d_space,space,totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, totalSize, cudaMemcpyHostToDevice);

    cudaCheckErrors("Copy mem");
#if PRINT_RESULTS
    Printer bytePrinter("device.data");
#endif
    printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

    for(i=0;i<NUM_OF_ITERATIONS;++i)
    {
    	simulation(d_params,d_space,d_result,type);
		swapPointers(d_space,d_result);

#if PRINT_RESULTS
		if(i % 2 == 0)
			cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);
		else
			cudaMemcpy(space, d_space, totalSize, cudaMemcpyDeviceToHost);
        bytePrinter.printIteration(space, i);
#else
        cudaThreadSynchronize();
#endif
    }
#if !PRINT_RESULTS
    if(i % 2 == 0)
		cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(space, d_space, totalSize, cudaMemcpyDeviceToHost);
#endif
    Timer::getInstance().stop("Device simulation time");
    printf("Simulation completed\n");
    Timer::getInstance().printResults();

    cudaFree(d_params);
    cudaFree(d_space);
    cudaFree(d_result);
    return space;
}

int main()
{
    bool hostSimulationOn = true;
    enum deviceSimulationType type = SURFACE;

    Fraction* hostOutputSpace,* deviceOutputSpace;

    initCuda();
    deviceOutputSpace = execDevice(type);

    if(hostSimulationOn)
    {
        hostOutputSpace = execHost();
        compare_results(hostOutputSpace,deviceOutputSpace);
        free(hostOutputSpace);
    }

    free(deviceOutputSpace);
    return 0;
}
