#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "customTimer.h"
#include "printer.h"
#include "Fraction.h"

#define RANDOM false
//#define PRINT_RESULTS false

Fraction* execHost(StartArgs args)
{
    Timer::getInstance().clear();
    int totalSize=sizeof(Fraction) * args.SIZE(), i;
    Printer* bytePrinter = NULL;
    void* space,*result= new Fraction[totalSize];
    auto params = initParams();
    if (NULL == result)
    {
        printf("Malloc problem!\n");
        exit(-1);
    }

    space = initSpace(args,RANDOM);
    if(args.print)
    	bytePrinter = new Printer("host.data",args);

    printf("Host simulation started\n");
    Timer::getInstance().start("Host simulation time");
    for(i=0;i<args.NUM_OF_ITERATIONS;++i)
    {
    	hostSimulation(args,&params,space,result);
		swapPointers(space,result);
		if(args.print)
			if(i % 2==0)
				bytePrinter->printIteration((Fraction*)space, i);
			else
				bytePrinter->printIteration((Fraction*)result, i);
    }
    Timer::getInstance().stop("Host simulation time");
    printf("Host simulation completed\n");
    Timer::getInstance().printResults();

    if(bytePrinter) delete bytePrinter;

    if(i % 2 == 0)
    {
    	free(result);
		return (Fraction*)space;
    }
    else
    {
    	free(space);
		return (Fraction*)result;
    }
}

Fraction* execDeviceSurface(StartArgs args,FluidParams* params,Fraction* space)
{
    float* floats =spaceToFloats(args,space);
    Printer* bytePrinter = NULL;

    // For float we could create a channel with:
    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc<float>();

    // Allocate memory in device
    cudaExtent extent = make_cudaExtent(args.X_SIZE*5,args.Y_SIZE,args.Z_SIZE);
    cudaArray* cuSpaceArray;
    cudaMalloc3DArray(&cuSpaceArray, &channelDesc,extent,cudaArraySurfaceLoadStore);
    cudaArray* cuResultArray;
    cudaMalloc3DArray(&cuResultArray, &channelDesc,extent,cudaArraySurfaceLoadStore);

    // Copy to device memory initial data
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)floats, args.X_SIZE*sizeof(float)*5, args.Y_SIZE, args.Z_SIZE);
    copyParams.dstArray = cuSpaceArray;
    copyParams.extent = extent;
    copyParams.kind	= cudaMemcpyHostToDevice;

    cudaMemcpy3D(&copyParams);

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

    if(args.print)
    	bytePrinter = new Printer("device.data",args);

    printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

    for(;i<args.NUM_OF_ITERATIONS;++i)
    {
    	simulationSurface(args,params,*(cudaSurfaceObject_t*)spaceObjPointer,*(cudaSurfaceObject_t*)resultObjPointer);
		swapPointers(spaceObjPointer,resultObjPointer);

		if(args.print)
		{
			copyParams = {0};
			copyParams.extent = extent;
			copyParams.dstPtr = make_cudaPitchedPtr((void*)floats, args.X_SIZE*sizeof(float)*5, args.Y_SIZE, args.Z_SIZE);
			copyParams.kind	= cudaMemcpyDeviceToHost;
			if(i % 2 == 0)
			{
				copyParams.srcArray = cuSpaceArray;
				cudaMemcpy3D(&copyParams);
			}
			else
			{
				copyParams.srcArray = cuResultArray;
				cudaMemcpy3D(&copyParams);
			}
			floatsToSpace(args,floats,space);
			bytePrinter->printIteration(space, i);
		}
    }
    if(!args.print)
    {
		copyParams = {0};
		copyParams.extent = extent;
		copyParams.dstPtr = make_cudaPitchedPtr((void*)floats, args.X_SIZE*sizeof(float)*5, args.Y_SIZE, args.Z_SIZE);
		copyParams.kind	= cudaMemcpyDeviceToHost;
		if(i % 2 == 0)
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

    floatsToSpace(args,floats,space);

    //free temporary memory
    free(floats);

    if(bytePrinter) delete bytePrinter;

    return space;
}

Fraction* execDevice(StartArgs args)
{
    Fraction* space = initSpace(args,RANDOM);
    Printer* bytePrinter =  NULL;
    FluidParams *d_params,params = initParams();
    cudaMalloc((void **)&d_params, sizeof(FluidParams));
    cudaMemcpy(d_params, &params, sizeof(FluidParams), cudaMemcpyHostToDevice);

    if(NULL==space)
        exit(-1);

    //DUE TO PROBLEMS WITH POINTERS AND SURFACE MEMORY OBJECTS THIS KIND OF SIMULATION IS THREATED SEPARATELY
    if(SURFACE==args.type)
        return execDeviceSurface(args, d_params,space);
    void *d_space,*d_result;
    int totalSize = sizeof(Fraction)*args.SIZE(), i;
    void *result = new Fraction[totalSize];

    cudaMalloc((void **)&d_space,totalSize);
    cudaMalloc((void **)&d_result,totalSize);
    cudaCheckErrors("Mallocs");
    cudaMemcpy(d_space,space,totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, totalSize, cudaMemcpyHostToDevice);

    cudaCheckErrors("Copy mem");
    if(args.print)
    	bytePrinter = new Printer("device.data",args);
    printf("Simulation started\n");
    Timer::getInstance().start("Device simulation time");

    for(i=0;i<args.NUM_OF_ITERATIONS;++i)
    {
    	simulation(args,d_params,d_space,d_result);
		swapPointers(d_space,d_result);

		if(args.print)
		{
			if(i % 2 == 0)
				cudaMemcpy(space, d_space, totalSize, cudaMemcpyDeviceToHost);
			else
				cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);
			bytePrinter->printIteration(space, i);
		}
		else
			cudaThreadSynchronize();
    }
    if(!args.print)
		if(i % 2 == 0)
			cudaMemcpy(space, d_space, totalSize, cudaMemcpyDeviceToHost);
		else
			cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);

    Timer::getInstance().stop("Device simulation time");
    printf("Simulation completed\n");
    Timer::getInstance().printResults();

    cudaFree(d_params);
    cudaFree(d_space);
    cudaFree(d_result);

    if(bytePrinter) delete bytePrinter;

    return space;
}

void printHelp()
{
	printf("hydro [options]\n"
			"--device <type>\t run simulation on GPU, types:\n"
			"\t\tGLOBAL, SURFACE, SHARED, SHARED_FOR\n"
			"--host\t run simulation on CPU\n"
			"--X <size>\t X map size\n"
			"--Y <size>\t Y map size\n"
			"--Z <size>\t Z map size\n"
			"--ITER <num>\t number of iterations\n"
			"--save\t\t save all iterations to file, default only last\n"
			"--help\t\t displays this massage\n"
			"if there is no arguments provided default values are:\n"
			"\thydro --device GLOBAL --X 100 --Y 100 --Z 100 --ITER 10\n");
}

StartArgs parsInputArguments(const int argc, char *argv[])
{
	bool error=false;
	StartArgs args;
	int x,y,z,iter_num;

	//default simulation settings
	args.NUM_OF_ITERATIONS = 10;
	args.X_SIZE = 100;
	args.Y_SIZE = 100;
	args.Z_SIZE = 100;
	args.host = false;
	args.type = GLOBAL;
	args.print = false;

	for(int i=1; i<argc; ++i)
	{
		if(strcmp(argv[i],"--device") == 0)
		{
			if((i+1)<argc && strcmp(argv[i+1],"GLOBAL") == 0)
				args.type = GLOBAL;
			else if((i+1)<argc && strcmp(argv[i+1],"SURFACE") == 0)
				args.type = SURFACE;
			else if((i+1)<argc && strcmp(argv[i+1],"SHARED") == 0)
				args.type = SHARED_3D_LAYER;
			else if((i+1)<argc && strcmp(argv[i+1],"SHARED_FOR") == 0)
				args.type = SHARED_3D_LAYER_FOR_IN;
			else
			{
				error = true;
				break;
			}
			++i;
		}
		else if(strcmp(argv[i],"--host") == 0)
		{
			args.host = true;
		}
		else if(strcmp(argv[i],"--X") == 0)
		{
			if((i+1)<argc && sscanf(argv[i+1],"%d",&x) == 1)
				args.X_SIZE = x;
			else
			{
				error = true;
				break;
			}
			++i;
		}
		else if(strcmp(argv[i],"--Y") == 0)
		{
			if((i+1)<argc && sscanf(argv[i+1],"%d",&y) == 1)
				args.Y_SIZE = y;
			else
			{
				error = true;
				break;
			}
			++i;
		}
		else if(strcmp(argv[i],"--Z") == 0)
		{
			if((i+1)<argc && sscanf(argv[i+1],"%d",&z) == 1)
				args.Z_SIZE = z;
			else
			{
				error = true;
				break;
			}
			++i;
		}
		else if(strcmp(argv[i],"--ITER") == 0)
		{
			if((i+1)<argc && sscanf(argv[i+1],"%d",&iter_num) == 1)
				args.NUM_OF_ITERATIONS = iter_num;
			else
			{
				error = true;
				break;
			}
			++i;
		}
		else if(strcmp(argv[i],"--save") == 0)
		{
			args.print = true;
		}
		else if(strcmp(argv[i],"--help") == 0)
		{
			printHelp();
			exit(EXIT_SUCCESS);
		}
		else
		{
			error=true;
			break;
		}
	}

	if(error)
	{
		printf("ERROR on parsing input arguments, starting simulation with default arguments\n");
		printHelp();
	}

	return args;
}

int main(int argc, char *argv[])
{
    Fraction* hostOutputSpace,* deviceOutputSpace;

    StartArgs args = parsInputArguments(argc,argv);
    initCuda();
    deviceOutputSpace = execDevice(args);

    if(args.host)
    {
        hostOutputSpace = execHost(args);
        compare_results(args,hostOutputSpace,deviceOutputSpace);
        free(hostOutputSpace);
    }

    free(deviceOutputSpace);
    return 0;
}
