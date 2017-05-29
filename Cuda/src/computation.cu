#include "computation.cuh"
#include  <cmath>
static int __constant__ D_X_SIZE = Factors::X_SIZE;
static int __constant__ D_Y_SIZE = Factors::Y_SIZE;
static int __constant__ D_Z_SIZE = Factors::Z_SIZE;
static int __constant__ D_SIZE   = Factors::P_COUNT;
static int __constant__ D_RANGE  = CudaParams::RANGE_TH;
static float __constant__ D_G      = Factors::GRAVITY;
static float __constant__ D_MASS_P = Factors::MASS_P;
static float __constant__ D_T	   = Factors::T;
void diffuse(Particle* data);
void addForces(Particle* data,int iter);
void advectParticles(Particle* data);
void updateVelocity(Particle* data);
void updateIndicies(Particle* data);
/*
 * For more info:
 * 	https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch38.html
 *  https://www.hindawi.com/journals/ijcgt/2015/417417/
 */

/*   To do (to increase performance):
 *      - each thread should have more calulation to compute ( Create FOR )
 *      -
 */

__device__ int id3dpitch(int x,int y,int z)
{
	return ((z*(D_X_SIZE *D_Y_SIZE ))+(y* D_X_SIZE) + x);
}

__global__ void updateIndecies(Particle* data, int* pCount)
{
	  int idx = threadIdx.x + blockDim.x*blockIdx.x;

	  if(idx < D_SIZE)
	  {
		  float4 pos = data->position[idx];
		  int newIndex = id3dpitch(pos.x,pos.y,pos.z); // do we care about math rounding?
	  }
}

/*  					Position advocation
 * Description:
 * 	Here we move particles according to the velocity and time period
 * We calculate equatation:
 * p(t + dt) = p(t) + u(t)dt
 * Area is parted with 3d blocks
 */

__global__ void advectParticles_kernel(Particle* data,float dt)
{
	  int idx = threadIdx.x + blockDim.x*blockIdx.x;

	  if(idx < D_SIZE)
	  {
		  float4 pos = data->position[idx];
		  float4 v   = data->velocity[idx];

		  pos.x += (dt * v.x);
		  pos.y += (dt * v.y);
		  pos.z += (dt * v.z);

		  data->position[idx] = pos;
	  }
}

__global__ void updateVelocity_kernel(Particle* data)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	if(idx< D_SIZE)
	{

		float4 v = data->velocity[idx];
		float4 f = data->externalForces[idx];
		v.x += D_T * f.x / D_MASS_P;
		v.y += D_T * f.y / D_MASS_P;
		v.z += D_T * f.z / D_MASS_P;

		data->velocity[idx] = v;
	}
}

__global__ void diffuse_kernel(Particle* data)
{


}

__global__ void addForces_kernel(Particle* data)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	float4 g;
	g.y = D_G; g.x= 0; g.z = 0;

	if(idx< D_SIZE)
	{
		data->externalForces[idx] = g;

		// TO DO
	}
}

void simulateFluids(Particle* data,int iter)
{
	//diffuse(data);
	addForces(data,iter);
	updateVelocity(data);
	advectParticles(data);
}


void diffuse(Particle* data)
{
	int blockNum = ceil(float(Factors::P_COUNT) / float(CudaParams::BLOCK_SIZE));
	diffuse_kernel<<<blockNum,CudaParams::BLOCK_SIZE>>>(data);
    cudaCheckErrors("advectDiffuse failed!");
}

void addForces(Particle* data,int iter)
{
	using namespace Factors;
	// For now only in 2D

	int blockNum = ceil(float(Factors::P_COUNT) / float(CudaParams::BLOCK_SIZE));
    addForces_kernel<<<blockNum,CudaParams::BLOCK_SIZE>>>(data);
    cudaCheckErrors("addForeces failed!");
}
void advectParticles(Particle* data)
{
	int blockNum = ceil(float(Factors::P_COUNT / CudaParams::BLOCK_SIZE));
	advectParticles_kernel<<<blockNum,CudaParams::BLOCK_SIZE>>>(data,Factors::T);
    cudaCheckErrors("advectParticles failed!");
}
void updateVelocity(Particle* data)
{
	int blockNum = ceil(float(Factors::P_COUNT) / float(CudaParams::BLOCK_SIZE));
	updateVelocity_kernel<<<blockNum,CudaParams::BLOCK_SIZE>>>(data);
    cudaCheckErrors("updateVelocity failed!");
}

int  IDX_3D(int x,int y,int z)
{
	return ((z*(Factors::X_SIZE*Factors::Y_SIZE))+
			(y* Factors::X_SIZE)+x);
}
