#include "exec.h"

Fraction* execDeviceSurface(StartArgs args, FluidParams* params, Fraction* space)
{
    float* floats = spaceToFloats(args, space);
    Printer* bytePrinter = NULL;

    // For float we could create a channel with:
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    // Allocate memory in device
    cudaExtent extent = make_cudaExtent(args.X_SIZE * 5, args.Y_SIZE, args.Z_SIZE);
    cudaArray* cuSpaceArray;
    cudaMalloc3DArray(&cuSpaceArray, &channelDesc, extent, cudaArraySurfaceLoadStore);
    cudaArray* cuResultArray;
    cudaMalloc3DArray(&cuResultArray, &channelDesc, extent, cudaArraySurfaceLoadStore);

    // Copy to device memory initial data
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)floats, args.X_SIZE * sizeof(float) * 5, args.Y_SIZE, args.Z_SIZE);
    copyParams.dstArray = cuSpaceArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&copyParams);

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    resDesc.res.array.array = cuSpaceArray;
    cudaSurfaceObject_t spaceSurfObj = 0;
    cudaCreateSurfaceObject(&spaceSurfObj, &resDesc);
    resDesc.res.array.array = cuResultArray;
    cudaSurfaceObject_t resultSurfObj = 0;
    cudaCreateSurfaceObject(&resultSurfObj, &resDesc);

    void *resultObjPointer = &resultSurfObj, *spaceObjPointer = &spaceSurfObj;
    int i = 0;

    if (args.print)
        bytePrinter = new Printer("device.data", args);

    printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

    for (; i<args.NUM_OF_ITERATIONS; ++i)
    {
        simulationSurface(args, params, *(cudaSurfaceObject_t*)spaceObjPointer, *(cudaSurfaceObject_t*)resultObjPointer);
        swapPointers(spaceObjPointer, resultObjPointer);

        if (args.print)
        {
            copyParams = { 0 };
            copyParams.extent = extent;
            copyParams.dstPtr = make_cudaPitchedPtr((void*)floats, args.X_SIZE * sizeof(float) * 5, args.Y_SIZE, args.Z_SIZE);
            copyParams.kind = cudaMemcpyDeviceToHost;
            if (i % 2 == 0)
            {
                copyParams.srcArray = cuSpaceArray;
                cudaMemcpy3D(&copyParams);
            }
            else
            {
                copyParams.srcArray = cuResultArray;
                cudaMemcpy3D(&copyParams);
            }
            floatsToSpace(args, floats, space);
            bytePrinter->printIteration(space, i);
        }
    }
    if (!args.print)
    {
        copyParams = { 0 };
        copyParams.extent = extent;
        copyParams.dstPtr = make_cudaPitchedPtr((void*)floats, args.X_SIZE * sizeof(float) * 5, args.Y_SIZE, args.Z_SIZE);
        copyParams.kind = cudaMemcpyDeviceToHost;
        if (i % 2 == 0)
        {
            copyParams.srcArray = cuSpaceArray;
            cudaMemcpy3D(&copyParams);
        }
        else
        {
            copyParams.srcArray = cuResultArray;
            cudaMemcpy3D(&copyParams);
        }
    }
    Timer::getInstance().stop("Device simulation time");
    printf("Simulation completed\n");
    Timer::getInstance().printResults();

    // Destroy surface objects
    cudaDestroySurfaceObject(spaceSurfObj);
    cudaDestroySurfaceObject(resultSurfObj);

    // Free device memory
    cudaFreeArray(cuSpaceArray);
    cudaFreeArray(cuResultArray);

    floatsToSpace(args, floats, space);

    //free temporary memory
    free(floats);

    if (bytePrinter) delete bytePrinter;

    return space;
}

Fraction* execDevice(StartArgs args)
{
    Fraction* space = initSpace(args);
    Printer* bytePrinter = NULL;
    FluidParams *d_params, params = initParams();
    cudaMalloc((void **)&d_params, sizeof(FluidParams));
    cudaMemcpy(d_params, &params, sizeof(FluidParams), cudaMemcpyHostToDevice);

    if (NULL == space)
        exit(-1);

    //DUE TO PROBLEMS WITH POINTERS AND SURFACE MEMORY OBJECTS THIS KIND OF SIMULATION IS THREATED SEPARATELY
    if (deviceSimulationType::SURFACE == args.type)
        return execDeviceSurface(args, d_params, space);
    void *d_space, *d_result;
    int totalSize = sizeof(Fraction)*args.SIZE(), i;
    void *result = new Fraction[totalSize];

    cudaMalloc((void **)&d_space, totalSize);
    cudaMalloc((void **)&d_result, totalSize);
    cudaCheckErrors("Mallocs");
    cudaMemcpy(d_space, space, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, totalSize, cudaMemcpyHostToDevice);

    cudaCheckErrors("Copy mem");
    if (args.print)
        bytePrinter = new Printer("device.data", args);
    printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

    for (i = 0; i<args.NUM_OF_ITERATIONS; ++i)
    {
        simulation(args, d_params, d_space, d_result);
        swapPointers(d_space, d_result);

        if (args.print)
        {
            if (i % 2 == 0)
                cudaMemcpy(space, d_space, totalSize, cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);
            bytePrinter->printIteration(space, i);
        }
        else
            cudaThreadSynchronize();
    }
    if (!args.print)
        if (i % 2 == 0)
            cudaMemcpy(space, d_space, totalSize, cudaMemcpyDeviceToHost);
        else
            cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);

    Timer::getInstance().stop("Device simulation time");
    printf("Simulation completed\n");
    Timer::getInstance().printResults();

    cudaFree(d_params);
    cudaFree(d_space);
    cudaFree(d_result);

    if (bytePrinter) delete bytePrinter;

    return space;
}
