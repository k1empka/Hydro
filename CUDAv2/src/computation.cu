#include "computation.cuh"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while(0);


__device__ float bilinterp(float* source, float x, float y,int xSize, int ySize)
{
	int x0,x1,y0,y1;
	float fx0,fx1,fy0,fy1;

	//Boundaries
	if(x > float(xSize)-2.0+0.5)
		x = float(xSize)-2.0+0.5;
	if(y > float(ySize)-2.0+0.5)
		y = float(ySize)-2.0+0.5;
	if(x < 0.5f)
		x = 0.5f;
	if(y < 0.5f)
		y = 0.5f;

	x0 = int(x);
	x1 = x0 + 1;
	y0 = int(y);
	y1 = y0 + 1;

	fx1 = (float)x - x0;
	fx0 = (float)1 - fx1;
	fy1 = (float)y - y0;
	fy0 = (float)1 - fy1;

	return 		   (float) fx0 * (fy0 * source[IDX_2D(x0,y0)]  +
						          fy1 * source[IDX_2D(x0,y1)]) +
					       fx1 * (fy0 * source[IDX_2D(x1,y0)]  +
					    		  fy1 * source[IDX_2D(x1,y1)]);
}

__global__ void advect(fraction* spaceData,fraction* resultData, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x > 1 && y > 1 && x<X_SIZE && y<Y_SIZE)
	{
		int id = IDX_2D(x,y);
		int newX = float(x) - dt * spaceData->Vx[id];
		int newY = float(y) - dt * spaceData->Vy[id];

		resultData->Vx[id] = bilinterp(spaceData->Vx,newX,newY,X_SIZE,Y_SIZE);
		resultData->Vy[id] = bilinterp(spaceData->Vy,newX,newY,X_SIZE,Y_SIZE);
		resultData->U[id]  = bilinterp(spaceData->U, newX,newY,X_SIZE,Y_SIZE);
	}
}


/*				Shared memory model
 * 				 ________________
 * 				|t1				|
 * 			____|t2_____________|_____
 * 		  |t1|t2|t1|t2|			|     |
 * 		  |	    |t2				|	  |
 * 		  |  	|				|	  |
 * 		  | 	|				|	  |
 * 		  |  	|				|	  |
 * 		  |  	|				|	  |
 * 		  |_____|_______________|_____|
 * 				|				|
 * 				|_______________|
 *
 * 			Border threads like t1,t2 copy also memory as described above
 */


__global__ void stepSh(fraction* spaceData,fraction* resultData)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	const int nCount = 2; //neighbours count
	extern __shared__ float shSpace[];
	shSpace[THX_3D(threadIdx.x,threadIdx.y,threadIdx.z)] = 0;
	shSpace[THX_3D(threadIdx.x+4,threadIdx.y+4,threadIdx.z+4)] = 0;

	if(x<X_SIZE && y<Y_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;
		int thx = threadIdx.x+2, thy = threadIdx.y+2,thz = threadIdx.z+2;
		int idx = IDX_3D(x,y,z);

		__syncthreads(); // wait for threads to fill whole shared memory

		shSpace[THX_3D(thx,thy,thz)] = space[IDX_3D(x,y,z)];

		if(threadIdx.x == 0 && x > 1)
		{
			shSpace[THX_2D(thx - 2,thy)] = space[IDX_2D(x-2,y)];
		}
		if(threadIdx.x == 1 && x > 2)
		{
			shSpace[THX_3D(thx - 2,thy,thz)] = space[IDX_3D(x-2,y,z)];
		}
		if(threadIdx.x == blockDim.x - nCount && x < X_SIZE - 2)
		{
			shSpace[THX_3D(thx + nCount,thy,thz)] = space[IDX_3D(x+nCount,y,z)];
		}
		if(threadIdx.x == blockDim.x - 1 && x < X_SIZE - 1)
		{
			shSpace[THX_3D(thx + nCount,thy,thz)] = space[IDX_3D(x+nCount,y,z)];
		}
		if(threadIdx.y == 0 && y > 1)
		{
			shSpace[THX_3D(thx,thy - nCount,thz)] = space[IDX_3D(x,y-nCount,z)];
		}
		if(threadIdx.y  == 1 && y > 2)
		{
			shSpace[THX_3D(thx,thy - nCount,thz)] = space[IDX_3D(x,y-nCount,z)];
		}
		if(threadIdx.y  == blockDim.y - nCount && y < Y_SIZE - 2)
		{
			shSpace[THX_3D(thx,thy + nCount,thz)] = space[IDX_3D(x,y+nCount,z)];
		}
		if(threadIdx.y  == blockDim.y - 1 && y < Y_SIZE - 1)
		{
			shSpace[THX_3D(thx,thy + nCount,thz)] = space[IDX_3D(x,y+nCount,z)];
		}
		if(threadIdx.z == 0 && z > 1)
		{
			shSpace[THX_3D(thx,thy,thz - nCount)] = space[IDX_3D(x,y,z - nCount)];
		}
		if(threadIdx.z  == 1 && z > 2)
		{
			shSpace[THX_3D(thx,thy,thz - nCount)] = space[IDX_3D(x,y,z - nCount)];
		}
		if(threadIdx.z  == blockDim.z - nCount && z < Z_SIZE - 2)
		{
			shSpace[THX_3D(thx,thy,thz + nCount)] = space[IDX_3D(x,y,z+nCount)];
		}
		if(threadIdx.z  == blockDim.z - 1 && z < Z_SIZE - 1)
		{
			shSpace[THX_3D(thx,thy,thz + nCount)] = space[IDX_3D(x,y,z+nCount)];
		}
		__syncthreads(); // wait for threads to fill whole shared memory

		result[idx]  = 0.7  * shSpace[THX_3D(thx,thy,thz)];

		result[idx] += 0.03 * shSpace[THX_3D(thx,thy-1,thz)];
		result[idx] += 0.02 * shSpace[THX_3D(thx,thy-2,thz)];
		result[idx] += 0.03 * shSpace[THX_3D(thx,thy+1,thz)];
		result[idx] += 0.02 * shSpace[THX_3D(thx,thy+2,thz)];
		result[idx] += 0.03 * shSpace[THX_3D(thx-1,thy,thz)];
		result[idx] += 0.02 * shSpace[THX_3D(thx-2,thy,thz)];
		result[idx] += 0.03 * shSpace[THX_3D(thx+1,thy,thz)];
		result[idx] += 0.02 * shSpace[THX_3D(thx+2,thy,thz)];
		result[idx] += 0.03 * shSpace[THX_3D(thx,thy,thz-1)];
		result[idx] += 0.02 * shSpace[THX_3D(thx,thy,thz-2)];
		result[idx] += 0.03 * shSpace[THX_3D(thx,thy,thz+1)];
		result[idx] += 0.02 * shSpace[THX_3D(thx,thy,thz+2)];

	}
}

__global__ void step(fraction* spaceData,fraction* resultData)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x<X_SIZE && y<Y_SIZE && z<Z_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;
		int idx = IDX_3D(x,y,z);

		result[idx] = 0.7*space[idx];

		if( (x+1) < X_SIZE )
			result[idx] +=.03 *space[IDX_3D(x+1,y,z)];
		if( (x-1) > 0 )
			result[idx] +=.03 *space[IDX_3D(x-1,y,z)];
		if( (y+1) < Y_SIZE )
			result[idx] +=.03 *space[IDX_3D(x,y+1,z)];
		if( (y-1) > 0 )
			result[idx] +=.03 *space[IDX_3D(x,y-1,z)];
		if( (z+1) < Z_SIZE )
			result[idx] +=.03 *space[IDX_3D(x,y,z+1)];
		if( (z-1) > 0 )
			result[idx] +=.03 *space[IDX_3D(x,y,z-1)];
		if( (x+2) < X_SIZE )
			result[idx] +=.02 *space[IDX_3D(x+2,y,z)];
		if( (x-2) > 0 )
			result[idx] +=.02 *space[IDX_3D(x-2,y,z)];
		if( (y+2) < Y_SIZE )
			result[idx] +=.02 *space[IDX_3D(x,y+2,z)];
		if( (y-2) > 0 )
			result[idx] +=.02 *space[IDX_3D(x,y-2,z)];
		if( (z+2) < Z_SIZE )
			result[idx] +=.02 *space[IDX_3D(x,y,z+2)];
		if( (z-2) > 0 )
			result[idx] +=.02 *space[IDX_3D(x,y,z-2)];
	}
}


void simulation(fraction* d_space,fraction* d_result)
{
	static dim3 threadsPerBlock(TH_IN_BLCK_X, TH_IN_BLCK_Y,TH_IN_BLCK_Z);
	static dim3 numBlocks(ceil(float(X_SIZE) / float(threadsPerBlock.x)),
						  ceil(float(Y_SIZE) / float(threadsPerBlock.y)),
						  ceil(float(Z_SIZE) / float(threadsPerBlock.z)));//threadsPerBlock.z=1;
	static int	shMemSize = sizeof(float) *
		(threadsPerBlock.x + 4) *
		(threadsPerBlock.y + 4) *
		(threadsPerBlock.z + 4); // each thread - each cell);
				// + boundaries threads need neighbours from other block

	//advect<<<numBlocks,threadsPerBlock>>>(d_space,d_result,DT);
	step<<<numBlocks, threadsPerBlock>>>(d_space,d_result);
	//stepSh<<<numBlocks, threadsPerBlock,shMemSize>>>(d_space,d_result);
    cudaCheckErrors("stepSh failed!");
}
