#include <helper_cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define NUM_OF_ITERATIONS 100
#define X_SIZE 100
#define Y_SIZE 100
#define NUM_OF_START_FRACTIONS 100
#define MAX_START_FORCE 100

struct fraction
{
	float U;
	//TODO more paramas
};

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
	fraction* space = (fraction*)malloc(X_SIZE*Y_SIZE*sizeof(fraction));

	if(NULL==space)
	{
		printf("memory allocation error\n");
		return NULL;
	}

	for(int i=0; i<Y_SIZE; ++i)
		for(int j=0; j<X_SIZE;++j)
		{
			space[i*Y_SIZE+j].U=0;
		}

	srand (time(NULL));

	for(int i=0; i<NUM_OF_START_FRACTIONS; ++i)
	{
		int x = (int)rand()%X_SIZE;
		int y = (int)rand()%Y_SIZE;

		space[y*X_SIZE+x].U=(float)(rand()%MAX_START_FORCE+1);
	}

	return space;
}

__global__ void step(fraction* space,fraction* result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x<X_SIZE && y<Y_SIZE)
	{
		result[y*X_SIZE+x].U=.7*space[y*X_SIZE+x].U;

		if( (y-1) > 0 )
			result[y*X_SIZE+x].U+=.05* space[(y-1)*X_SIZE+x].U;
		if( (y-2) > 0 )
			result[y*X_SIZE+x].U+=.025*space[(y-2)*X_SIZE+x].U;
		if( (y+1) < Y_SIZE )
			result[y*X_SIZE+x].U+=.05* space[(y+1)*X_SIZE+x].U;
		if( (y+2) < Y_SIZE )
			result[y*X_SIZE+x].U+=.025*space[(y+2)*X_SIZE+x].U;
		if( (x-1) > 0 )
			result[y*X_SIZE+x].U+=.05* space[(y)*X_SIZE+x-1].U;
		if( (x-2) > 0 )
			result[y*X_SIZE+x].U+=.025*space[(y)*X_SIZE+x-2].U;
		if( (x+1) < X_SIZE )
			result[y*X_SIZE+x].U+=.05* space[(y)*X_SIZE+x+1].U;
		if( (x+2) < X_SIZE )
			result[y*X_SIZE+x].U+=.025*space[(y)*X_SIZE+x+2].U;

	}
}

__global__ void copySpace(fraction* from,fraction* to)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x<X_SIZE && y<Y_SIZE)
	{
		to[y*X_SIZE+x].U=from[y*X_SIZE+x].U;
	}
}


void printHeader(FILE* f)
{
	fprintf(f,"%d %d %d\n",X_SIZE,Y_SIZE,NUM_OF_ITERATIONS);
}

void printIteration(FILE* f,fraction* space, int iter)
{
	fprintf(f,"ITER_%d\n",iter);

	for(int i=0; i<Y_SIZE;++i)
		for(int j=0; j<X_SIZE;++j)
		{
			if(space[i*X_SIZE+j].U) fprintf(f,"%d %d %f %f\n",j,i,space[i*X_SIZE+j].U,space[i*X_SIZE+j].U);
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
	if(NULL==space) return -1;

	int N = X_SIZE * Y_SIZE;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	fraction *d_space,*d_result;
	int totalSize=X_SIZE*Y_SIZE*sizeof(fraction);
	cudaMalloc((void **)&d_space,totalSize);
	cudaMalloc((void **)&d_result,totalSize);
	cudaMemcpy(d_space,space,totalSize, cudaMemcpyHostToDevice);

	FILE* f = initOutputFile();

	printHeader(f);

	printf("Simulation started\n");
	for(int i=0;i<NUM_OF_ITERATIONS;++i)
	{
		step<<<numBlocks, threadsPerBlock>>>(d_space,d_result);
		copySpace<<<numBlocks, threadsPerBlock>>>(d_result,d_space);
		cudaMemcpy(space,d_space,totalSize, cudaMemcpyDeviceToHost);
		printIteration(f,space,i);
	}
	printf("Simulation complted\n");

	cudaFree(d_space);
	cudaFree(d_result);
	free(space);

	fclose(f);

	return 0;
}
