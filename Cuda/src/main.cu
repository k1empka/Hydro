#include <iostream>
#include <ctime>
#include <helper_cuda.h>

#include "particle.h"
#include "computation.cuh"
#include "parser.h"
#include "Timer.h"

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

	const int totalS = P_COUNT;
	data = new Particle();
	memset(data->position,0,sizeof(float4) * totalS);
	memset(data->velocity,0,sizeof(float4) * totalS);

	for(int i = 0;i < totalS; ++i)
	{
		float4 pos;
		pos.x = ((rand() % INIT_FLUID_W ));
		pos.y = ((rand() % INIT_FLUID_W ));
		pos.z = ((rand() % INIT_FLUID_W ));
		data->position[i] = pos;
		data->velocity[i].x = float(X_SIZE / 2. - rand() % X_SIZE) * 0.05f;
		data->velocity[i].y = float(Y_SIZE / 2. - rand() % Y_SIZE) * 0.01f;
		data->velocity[i].z = float(Z_SIZE / 2. - rand() % Z_SIZE) * 0.01f;
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
    parser.writeIterToFile2D(h_data,0);

    for(int i = 0; i < Factors::ITERATION_NUM; ++i)
    {
        Timer::getInstance().start("Simulation Time");
    	simulateFluids(d_data,i);
        cudaCheckErrors("simulation fluid failed!");
        cudaMemcpy(h_data,d_data,totalSize, cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy to host failed");
        Timer::getInstance().stop("Simulation Time");
        //parser.writeIterToFile2D(h_data,i+1);
    }

    cudaFree(d_data);
    delete h_data;
    std::cout << "Simulation succeed!" << std::endl;
    Timer::getInstance().printResults();
	return 0;
}
