#include "computation.cuh"

static int __constant__ D_X_SIZE = Factors::X_SIZE;
static int __constant__ D_Y_SIZE = Factors::Y_SIZE;
static int __constant__ D_Z_SIZE = Factors::Z_SIZE;
static int __constant__ D_SIZE   = Factors::TOTAL_SIZE;
static int __constant__ D_RANGE  = CudaParams::RANGE_TH;

void advectVelocity(Particle* data);
void diffuse(Particle* data);
void addForces();
void advectParticles(Particle* data);
void updateVelocity(Particle* data);

/*
 * For more info:
 * 	https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch38.html
 *  https://www.hindawi.com/journals/ijcgt/2015/417417/
 */

/*   To do (to increase performance):
 *      - each thread should have more calulation to compute ( Create FOR )
 *      -
 */

void cudaKernelsConfig(dim3& grid,dim3& block)
{
	using namespace Factors;
	using namespace CudaParams;
    block.x = BLCK_X;
    block.y = BLCK_Y;
    block.z = BLCK_Z;
	grid.x = X_SIZE/BLCK_X;
	grid.x = Y_SIZE/BLCK_Y;
	grid.x = Z_SIZE/BLCK_Z;
}


/*  					Position advocation
 * Description:
 * 	Here we move particles according to the velocity and time period
 * We calculate equatation:
 * p(t + dt) = p(t) + u(t)dt
 * Area is parted with 3d blocks
 */

__device__ int id3dpitch(int x,int y,int z)
{
	return ((z*(D_X_SIZE *D_Y_SIZE ))+(y* D_X_SIZE) + x);
}

__global__ void advectParticles_kernel(Particle* data,float dt)
{
	  int idx = threadIdx.x + blockDim.x*blockIdx.x;
	  int idy = threadIdx.y + blockDim.y*blockIdx.y;
	  int idz = threadIdx.z + blockDim.z* blockIdx.z;

	  int gId = idz * D_X_SIZE *  D_Y_SIZE + (idy* D_X_SIZE) + idx;

	  if(gId < D_SIZE)
	  {
		  float4 pos = data->position[gId];
		  float4 v   = data->velocity[gId];

		  pos.x += (dt * v.x);
		  pos.y += (dt * v.y);
		  pos.z += (dt * v.z);

		  data->position[gId] = pos;
	  }
}
/*  					Velocity advocation
 * Description:
 * We calculate equatation:
 * q(x, t + dt) = q(x - u(x,t)dt,t)
 * Area is parted with 3d blocks
 */

__global__ void advectVelocity_kernel(Particle* data,float dt)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int idy = threadIdx.y + blockDim.y*blockIdx.y;
	int idz = threadIdx.z + blockDim.z*blockIdx.z;

	int gId = id3dpitch(idx,idy,idz);

	if(gId < D_SIZE)
	{
		float4 v   = data->velocity[gId];
		float4 pos;
		pos.x = (idx + 0.5f) - (dt * v.x);
		pos.y = (idy + 0.5f) - (dt * v.y);
		pos.z = (idz + 0.5f) - (dt * v.z);

		int newGId = id3dpitch((int)pos.x,(int)pos.y,(int)pos.z);
		data->velocityTmp[newGId] = v;
	}
}

__global__ void updateVelocity_kernel(Particle* data)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int idy = threadIdx.y + blockDim.y*blockIdx.y;
	int idz = threadIdx.z + blockDim.z*blockIdx.z;

	int gId = id3dpitch(idx,idy,idz);

	if(gId < D_SIZE)
	{
		float4 v   = data->velocityTmp[gId];
		data->velocity[gId] = v;
	}
}

__global__ void diffuse_kernel(Particle* data)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int idy = threadIdx.y + blockDim.y*blockIdx.y;
	int idz = threadIdx.z + blockDim.z*blockIdx.z;
	int gId = id3dpitch(idx,idy,idz);

	//TO DO

}

void simulateFluids(Particle* data)
{
	advectVelocity(data);
	diffuse(data);
	updateVelocity(data);
	advectParticles(data);
}

void advectVelocity(Particle* data)
{
    dim3 block,grid;
    cudaKernelsConfig(grid,block);
	advectVelocity_kernel<<<grid,block>>>(data,Factors::T);
    cudaCheckErrors("advectParticles failed!");
}

void diffuse(Particle* data)
{
    dim3 block,grid;
    cudaKernelsConfig(grid,block);
	diffuse_kernel<<<grid,block>>>(data);
    cudaCheckErrors("advectParticles failed!");
}

void addForces()
{

}
void advectParticles(Particle* data)
{
    dim3 block,grid;
    cudaKernelsConfig(grid,block);
	advectParticles_kernel<<<grid,block>>>(data,Factors::T);
    cudaCheckErrors("advectParticles failed!");
}
void updateVelocity(Particle* data)
{
    dim3 block,grid;
    cudaKernelsConfig(grid,block);
	updateVelocity_kernel<<<grid,block>>>(data);
    cudaCheckErrors("updateVelocity failed!");
}

int  IDX_3D(int x,int y,int z)
{
	return ((z*(Factors::X_SIZE*Factors::Y_SIZE))+
			(y* Factors::X_SIZE)+x);
}
