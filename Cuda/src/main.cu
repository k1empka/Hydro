#include <iostream>
#include <ctime>
#include <helper_cuda.h>

#include "particle.h"
#include "computation.cuh"
#include "parser.h"

#define ITERATION_NUM 1


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

Particle* initData()
{
	Particle* data;
	using namespace Factors;
	srand(time(NULL));

	auto startPos = [](int size)
	{
		return size / 2 - INIT_FLUID_W;
	};
	auto endPos = [](int size,int start)
	{
		return start + INIT_FLUID_W * 2;
	};

	const int totalS = TOTAL_SIZE;
	data = new Particle();
	memset(data->position,0,sizeof(float4) * totalS);
	memset(data->velocity,0,sizeof(float4) * totalS);

	int sx = startPos(X_SIZE), sy = startPos(Y_SIZE), sz = startPos(Z_SIZE);
	int ex = endPos(X_SIZE,sx), ey = endPos(Y_SIZE,sy), ez =endPos(Z_SIZE,sz);
	for(int i = sx;i < ex; ++i)
		for(int j = sy;j < ey; ++j)
			for(int k = sz;k < ez; ++k)
			{
				int idx = IDX_3D(i,j,k);
				data->position[idx].x = (sx+(rand() % ex));
				data->position[idx].y = (sy+(rand() % ey));
				data->position[idx].z = (sz+(rand() % ez));
				data->velocity[idx].x = (X_SIZE/2 - (rand() % X_SIZE)) * 0.05f;
				data->velocity[idx].y = (Y_SIZE/2 - (rand() % Y_SIZE)) * 0.05f;
				data->velocity[idx].z = (Z_SIZE/2 - (rand() % Z_SIZE)) * 0.05f;
			}
	return data;
}


int main(int argc,char* argv[])
{
	//TO DO: reading paths from args

	/*if(argc > 1)
	{
		parseEntryData(argv[1]);
	}*/
	initCuda();
	Parser parser("","results.txt");
	Particle* h_data = initData();
	Particle* d_data;

	size_t totalSize =  sizeof(Particle);
    cudaMalloc((void **)&d_data,totalSize);
    cudaCheckErrors("cudaMalloc failed");
    cudaMemcpy(d_data,h_data,totalSize, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy to device failed");

    std::cout << "Simulation started!" << std::endl;

    for(int i = 0; i < ITERATION_NUM; ++i)
    {
    	simulateFluids(d_data);
        cudaCheckErrors("simulation fluid failed!");
        cudaMemcpy(h_data,d_data,totalSize, cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy to host failed");
        parser.writeIterToFile2D(h_data,i);
    }
    cudaFree(d_data);
    std::cout << "Simulation succeed!" << std::endl;
	return 0;
}
