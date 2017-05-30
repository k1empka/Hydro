#include <helper_cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "computation.cuh"

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
    if(nDevices > 0)
        cudaSetDevice(nDevices - 1); //Dla mnie bo mam SLI;
}

fraction* initSpace()
{
	fraction* space = (fraction*)malloc(sizeof(fraction));

	if(NULL==space)
	{
		printf("memory allocation error\n");
		return NULL;
	}

	for(int y=0; y<Y_SIZE; ++y)
		for(int x=0; x<X_SIZE;++x)
		{
			space->Vx[IDX_2D(x,y)]=0.;
			space->Vy[IDX_2D(x,y)]=0.;
			space->U[IDX_2D(x,y)]=0.;
		}

	srand (time(NULL));

	for(int i=0; i<NUM_OF_START_FRACTIONS; ++i)
	{
		int x = (int)rand()%X_SIZE;
		int y = (int)rand()%Y_SIZE;

		space->U[y*X_SIZE+x]=(float)(rand()%MAX_START_FORCE + 1);
	}

	return space;
}

void printHeader(FILE* f)
{
	fprintf(f,"%d %d %d\n",X_SIZE,Y_SIZE,NUM_OF_ITERATIONS);
}

void printIteration(FILE* f,fraction* space, int iter)
{
	fprintf(f,"ITER_%d\n",iter);

	for(int y=0; y<Y_SIZE;++y)
		for(int x=0; x<X_SIZE;++x)
		{
			if(space->U[y*X_SIZE+x] != 0)
				fprintf(f,"%d %d %f %f\n",x,y,space->U[y*X_SIZE+x],space->U[y*X_SIZE+x]);
		}
}

FILE* initOutputFile()
{
	char filename[100];
	sprintf(filename,"%luresult",(unsigned long)time(NULL));
	FILE *f = fopen(filename, "w");
	if (f == NULL)
	{
	    printf("Error opening file!\n");
	    exit(1);
	}
	return f;
}

int main()
{
	initCuda();

	fraction* space = initSpace();

	if(NULL==space)
		return -1;

	fraction *d_space,*d_result;
	int totalSize=sizeof(fraction);
	cudaMalloc((void **)&d_space,totalSize);
	cudaMalloc((void **)&d_result,totalSize);
	cudaMemcpy(d_space,space,totalSize, cudaMemcpyHostToDevice);

	FILE* f = initOutputFile();

	printHeader(f);

	printf("Simulation started\n");
	for(int i=0;i<NUM_OF_ITERATIONS;++i)
	{
		simulation(d_space,d_result);
		cudaMemcpy(space,d_space,totalSize, cudaMemcpyDeviceToHost);
		printIteration(f,space,i);
	}
	printf("Simulation completed\n");

	cudaFree(d_space);
	cudaFree(d_result);
	free(space);

	fclose(f);

	return 0;
}
