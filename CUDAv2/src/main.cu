#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "exec.h"

void printHelp()
{
	printf("hydro [options]\n"
			"--device <type>\trun simulation on GPU, types:\n"
			"\t\tGLOBAL, SURFACE, SHARED, SHARED_FOR\n"
			"--host\t\trun simulation on CPU\n"
			"--X <size>\tX space size\n"
			"--Y <size>\tY space size\n"
			"--Z <size>\tZ space size\n"
			"--ITER <num>\tnumber of iterations\n"
			"--save\t\tsave all iterations to file, default only last\n"
			"--help\t\tdisplays this massage\n"
			"--random\tstarting space values are random\n"
			"if there is no arguments provided default values are:\n"
			"\thydro --device GLOBAL --X 100 --Y 100 --Z 100 --ITER 10\n");
}

StartArgs parsInputArguments(const int argc, char *argv[])
{
	bool error=false;
	StartArgs args;
	int x,y,z,iter_num;

	//default simulation settings
	args.NUM_OF_ITERATIONS = 50;
	args.X_SIZE = 100;
	args.Y_SIZE = 100;
	args.Z_SIZE = 100;
	args.host = hostSimulationType::NONE;
	args.type = deviceSimulationType::NONE;
	args.print = false;
	args.random = false;

	for(int i=1; i<argc; ++i)
	{
		if(strcmp(argv[i],"--device") == 0)
		{
			if((i+1)<argc && strcmp(argv[i+1],"GLOBAL") == 0)
				args.type = deviceSimulationType::GLOBAL;
			else if((i+1)<argc && strcmp(argv[i+1],"SURFACE") == 0)
				args.type = deviceSimulationType::SURFACE;
			else if((i+1)<argc && strcmp(argv[i+1],"SHARED") == 0)
				args.type = deviceSimulationType::SHARED_3D_LAYER;
			else if((i+1)<argc && strcmp(argv[i+1],"SHARED_FOR") == 0)
				args.type = deviceSimulationType::SHARED_3D_LAYER_FOR_IN;
			else
                args.type = deviceSimulationType::NONE;
			
			++i;
		}
		else if(strcmp(argv[i],"--host") == 0)
		{
            if ((i + 1)<argc && strcmp(argv[i + 1], "SPACE") == 0)
                args.host = hostSimulationType::SPACE;
            else if ((i + 1)<argc && strcmp(argv[i + 1], "OCTREE") == 0)
                args.host = hostSimulationType::OCTREE;
            else
                args.host = hostSimulationType::NONE;
		}
		else if(strcmp(argv[i],"--random") == 0)
		{
            args.random = true;
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
    if (args.type != deviceSimulationType::NONE)
    {
        initCuda();
        deviceOutputSpace = execDevice(args);
    }

    if(args.host != hostSimulationType::NONE)
    {
        hostOutputSpace = execHost(args);
        free(hostOutputSpace);
    }
    if (args.type != deviceSimulationType::NONE && args.host != hostSimulationType::NONE)
        compare_results(args, hostOutputSpace, deviceOutputSpace);

    return 0;
}
