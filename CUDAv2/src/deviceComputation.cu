#include "computation.h"
#include <helper_cuda.h>
#include <mutex>


/*
__device__ float bilinearinterpolation(float* source, float x, float y,int xSize, int ySize)
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
*/

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


__global__ void stepShared3DCube(fraction* spaceData,fraction* resultData)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	const int nCount = 2; //neighbours count
	extern __shared__ float shSpace[];
	shSpace[THX_3D(threadIdx.x,threadIdx.y,threadIdx.z)] = 0;
	shSpace[THX_3D(threadIdx.x+4,threadIdx.y+4,threadIdx.z+4)] = 0;

	if(x<X_SIZE && y<Y_SIZE && z<Z_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;
		const int thx = threadIdx.x+2, thy = threadIdx.y+2,thz = threadIdx.z+2;
		const int idx = IDX_3D(x,y,z);

		__syncthreads(); // wait for threads to fill whole shared memory

		shSpace[THX_3D(thx,thy,thz)] = space[IDX_3D(x,y,z)];

		if(threadIdx.x == 0 && x > 1)
		{
			shSpace[THX_3D(thx - 2,thy,thz)] = space[THX_3D(x-2,y,z)];
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
		float resultCell = 0.7  * shSpace[THX_3D(thx,thy,thz)];


		resultCell += 0.03 * shSpace[THX_3D(thx,thy-1,thz)];
		resultCell += 0.02 * shSpace[THX_3D(thx,thy-2,thz)];
		resultCell += 0.03 * shSpace[THX_3D(thx,thy+1,thz)];
		resultCell += 0.02 * shSpace[THX_3D(thx,thy+2,thz)];
		resultCell += 0.03 * shSpace[THX_3D(thx-1,thy,thz)];
		resultCell += 0.02 * shSpace[THX_3D(thx-2,thy,thz)];
		resultCell += 0.03 * shSpace[THX_3D(thx+1,thy,thz)];
		resultCell += 0.02 * shSpace[THX_3D(thx+2,thy,thz)];
		resultCell += 0.03 * shSpace[THX_3D(thx,thy,thz-1)];
		resultCell += 0.02 * shSpace[THX_3D(thx,thy,thz-2)];
		resultCell += 0.03 * shSpace[THX_3D(thx,thy,thz+1)];
		resultCell += 0.02 * shSpace[THX_3D(thx,thy,thz+2)];

		result[idx]  = resultCell;
	}
}

__global__ void stepShared3DLayer(fraction* spaceData,fraction* resultData,int z)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ float shSpace[];
	shSpace[THX_2D(threadIdx.x,threadIdx.y)] = 0;
	shSpace[THX_2D(threadIdx.x+4,threadIdx.y+4)] = 0;

	if(x<X_SIZE && y<Y_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;
		const int thx = threadIdx.x+2, thy = threadIdx.y+2;
		const int idx = IDX_3D(x,y,z);

		__syncthreads(); // wait for threads to fill whole shared memory

		shSpace[THX_2D(thx,thy)] = space[IDX_3D(x,y,z)];

		if(threadIdx.x == 0 && x > 1)
		{
			shSpace[THX_2D(thx - 2,thy)] = space[IDX_3D(x-2,y,z)];
		}
		if(threadIdx.x == 1 && x > 2)
		{
			shSpace[THX_2D(thx - 2,thy)] = space[IDX_3D(x-2,y,z)];
		}
		if(threadIdx.x == blockDim.x - 2 && x < X_SIZE - 2)
		{
			shSpace[THX_2D(thx + 2,thy)] = space[IDX_3D(x+2,y,z)];
		}
		if(threadIdx.x == blockDim.x - 1 && x < X_SIZE - 1)
		{
			shSpace[THX_2D(thx + 2,thy)] = space[IDX_3D(x+2,y,z)];
		}
		if(threadIdx.y == 0 && y > 1)
		{
			shSpace[THX_2D(thx,thy - 2)] = space[IDX_3D(x,y-2,z)];
		}
		if(threadIdx.y  == 1 && y > 2)
		{
			shSpace[THX_2D(thx,thy - 2)] = space[IDX_3D(x,y-2,z)];
		}
		if(threadIdx.y  == blockDim.y - 2 && y < Y_SIZE - 2)
		{
			shSpace[THX_2D(thx,thy + 2)] = space[IDX_3D(x,y+2,z)];
		}
		if(threadIdx.y  == blockDim.y - 1 && y < Y_SIZE - 1)
		{
			shSpace[THX_2D(thx,thy + 2)] = space[IDX_3D(x,y+2,z)];
		}

		__syncthreads(); // wait for threads to fill whole shared memory

		// Calculate cell  with data from shared memory (Layer part x,y)
		float resultCell;
		resultCell  = 0.7  * shSpace[THX_2D(thx,thy)];

		resultCell += 0.03 * shSpace[THX_2D(thx,thy-1)];
		resultCell += 0.02 * shSpace[THX_2D(thx,thy-2)];
		resultCell += 0.03 * shSpace[THX_2D(thx,thy+1)];
		resultCell += 0.02 * shSpace[THX_2D(thx,thy+2)];
		resultCell += 0.03 * shSpace[THX_2D(thx-1,thy)];
		resultCell += 0.02 * shSpace[THX_2D(thx-2,thy)];
		resultCell += 0.03 * shSpace[THX_2D(thx+1,thy)];
		resultCell += 0.02 * shSpace[THX_2D(thx+2,thy)];

		// Calculate the rest of data (Cross Z asix)
		if( (z+2) < Z_SIZE )
			resultCell += 0.02 *space[IDX_3D(x,y,z+2)];
		if( (z-2) >= 0 )
			resultCell += 0.02 *space[IDX_3D(x,y,z-2)];
		if( (z+1) < Z_SIZE )
			resultCell += 0.03 *space[IDX_3D(x,y,z+1)];
		if( (z-1) >= 0 )
			resultCell += 0.03 *space[IDX_3D(x,y,z-1)];

		result[idx] = resultCell;
	}
}

__global__ void stepShared3DLayerForIn(fraction* spaceData,fraction* resultData)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int nCount = 2; //neighbours count
	extern __shared__ float shSpace[];
	shSpace[THX_2D(threadIdx.x,threadIdx.y)] = 0;
	shSpace[THX_2D(threadIdx.x+4,threadIdx.y+4)] = 0;

	if(x<X_SIZE && y<Y_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;
		const int thx = threadIdx.x+2, thy = threadIdx.y+2;

		for(int z=0; z<Z_SIZE; ++z)
		{
			const int idx = IDX_3D(x,y,z);

			shSpace[THX_2D(thx,thy)] = space[IDX_3D(x,y,z)];

			if(threadIdx.x == 0 && x > 1)
			{
				shSpace[THX_2D(thx - 2,thy)] = space[IDX_3D(x-2,y,z)];
			}
			if(threadIdx.x == 1 && x > 2)
			{
				shSpace[THX_2D(thx - 2,thy)] = space[IDX_3D(x-2,y,z)];
			}
			if(threadIdx.x == blockDim.x - nCount && x < X_SIZE - 2)
			{
				shSpace[THX_2D(thx + nCount,thy)] = space[IDX_3D(x+nCount,y,z)];
			}
			if(threadIdx.x == blockDim.x - 1 && x < X_SIZE - 1)
			{
				shSpace[THX_2D(thx + nCount,thy)] = space[IDX_3D(x+nCount,y,z)];
			}
			if(threadIdx.y == 0 && y > 1)
			{
				shSpace[THX_2D(thx,thy - nCount)] = space[IDX_3D(x,y-nCount,z)];
			}
			if(threadIdx.y  == 1 && y > 2)
			{
				shSpace[THX_2D(thx,thy - nCount)] = space[IDX_3D(x,y-nCount,z)];
			}
			if(threadIdx.y  == blockDim.y - nCount && y < Y_SIZE - 2)
			{
				shSpace[THX_2D(thx,thy + nCount)] = space[IDX_3D(x,y+nCount,z)];
			}
			if(threadIdx.y  == blockDim.y - 1 && y < Y_SIZE - 1)
			{
				shSpace[THX_2D(thx,thy + nCount)] = space[IDX_3D(x,y+nCount,z)];
			}

			__syncthreads(); // wait for threads to fill whole shared memory

			float resultCell =  0.7  * shSpace[THX_2D(thx,thy)];
			// Calculate cell  with data from shared memory (Layer part x,y)

			resultCell += 0.03 * shSpace[THX_2D(thx,thy-1)];
			resultCell += 0.02 * shSpace[THX_2D(thx,thy-2)];
			resultCell += 0.03 * shSpace[THX_2D(thx,thy+1)];
			resultCell += 0.02 * shSpace[THX_2D(thx,thy+2)];
			resultCell += 0.03 * shSpace[THX_2D(thx-1,thy)];
			resultCell += 0.02 * shSpace[THX_2D(thx-2,thy)];
			resultCell += 0.03 * shSpace[THX_2D(thx+1,thy)];
			resultCell += 0.02 * shSpace[THX_2D(thx+2,thy)];

			// Calculate the rest of data (Cross Z asix)
			if( (z+2) < Z_SIZE )
				resultCell +=.02 *space[IDX_3D(x,y,z+2)];
			if( (z-2) >= 0 )
				resultCell +=.02 *space[IDX_3D(x,y,z-2)];
			if( (z+1) < Z_SIZE )
				resultCell +=.03 *space[IDX_3D(x,y,z+1)];
			if( (z-1) >= 0 )
				resultCell +=.03 *space[IDX_3D(x,y,z-1)];

			result[idx]  = resultCell;
		}

	}
}

__global__ void stepSharedForIn(fraction* spaceData,fraction* resultData)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int nCount = 2; //neighbours count
	extern __shared__ float shSpace[];
	shSpace[THX_3D(threadIdx.x,threadIdx.y,threadIdx.z)] = 0;
	shSpace[THX_3D(threadIdx.x+4,threadIdx.y+4,threadIdx.z+4)] = 0;

	if(x<X_SIZE && y<Y_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;
		int z = 0;
		int thz = z+2;
		const int thx = threadIdx.x+2,thy = threadIdx.y+2;
		const int idx = IDX_3D(x,y,z);

		//load initial data
		shSpace[THX_3D(thx,thy,thz)] = space[IDX_3D(x,y,z)];

		if(threadIdx.x == 0 && x > 1)
		{
			shSpace[THX_3D(thx - 2,thy,thz)] = space[IDX_3D(x-2,y,z)];
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

		for(; z < Z_SIZE; ++z)
		{
			thz = z+2;

			//now we will load only portion of data to shared memory
			if(threadIdx.x == 0 && x > 1)
			{
				shSpace[THX_3D(thx - 2,thy,thz)] = space[IDX_3D(x-2,y,z)];
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
			if(threadIdx.z  == blockDim.z - 1 && z < Z_SIZE - 1)
			{
				shSpace[THX_3D(thx,thy,thz + nCount)] = space[IDX_3D(x,y,z+nCount)];
			}

			__syncthreads(); // wait for threads to fill whole shared memory

			float resultCell = 0.7  * shSpace[THX_3D(thx,thy,thz)];

			resultCell += 0.03 * shSpace[THX_3D(thx,thy-1,thz)];
			resultCell += 0.02 * shSpace[THX_3D(thx,thy-2,thz)];
			resultCell += 0.03 * shSpace[THX_3D(thx,thy+1,thz)];
			resultCell += 0.02 * shSpace[THX_3D(thx,thy+2,thz)];
			resultCell += 0.03 * shSpace[THX_3D(thx-1,thy,thz)];
			resultCell += 0.02 * shSpace[THX_3D(thx-2,thy,thz)];
			resultCell += 0.03 * shSpace[THX_3D(thx+1,thy,thz)];
			resultCell += 0.02 * shSpace[THX_3D(thx+2,thy,thz)];
			resultCell += 0.03 * shSpace[THX_3D(thx,thy,thz-1)];
			resultCell += 0.02 * shSpace[THX_3D(thx,thy,thz-2)];
			resultCell += 0.03 * shSpace[THX_3D(thx,thy,thz+1)];
			resultCell += 0.02 * shSpace[THX_3D(thx,thy,thz+2)];

			result[idx]  = resultCell;
		}
	}
}

__global__ void stepGlobal(fraction* spaceData,fraction* resultData)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x<X_SIZE && y<Y_SIZE && z<Z_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;
		const int idx = IDX_3D(x,y,z);

		float resultCell = 0.7*space[idx];

		if( (x+1) < X_SIZE )
			resultCell +=.03 *space[IDX_3D(x+1,y,z)];
		if( (x-1) >= 0 )
			resultCell +=.03 *space[IDX_3D(x-1,y,z)];
		if( (y+1) < Y_SIZE )
			resultCell +=.03 *space[IDX_3D(x,y+1,z)];
		if( (y-1) >= 0 )
			resultCell +=.03 *space[IDX_3D(x,y-1,z)];
		if( (z+1) < Z_SIZE )
			resultCell +=.03 *space[IDX_3D(x,y,z+1)];
		if( (z-1) >= 0 )
			resultCell +=.03 *space[IDX_3D(x,y,z-1)];
		if( (x+2) < X_SIZE )
			resultCell +=.02 *space[IDX_3D(x+2,y,z)];
		if( (x-2) >= 0 )
			resultCell +=.02 *space[IDX_3D(x-2,y,z)];
		if( (y+2) < Y_SIZE )
			resultCell +=.02 *space[IDX_3D(x,y+2,z)];
		if( (y-2) >= 0 )
			resultCell +=.02 *space[IDX_3D(x,y-2,z)];
		if( (z+2) < Z_SIZE )
			resultCell +=.02 *space[IDX_3D(x,y,z+2)];
		if( (z-2) >= 0 )
			resultCell +=.02 *space[IDX_3D(x,y,z-2)];

		result[idx] = resultCell;
	}
}

__global__ void stepSurface(cudaSurfaceObject_t spaceData,cudaSurfaceObject_t resultData)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x<X_SIZE && y<Y_SIZE && z<Z_SIZE)
	{
		const int idx = IDX_3D(x,y,z);

		float data;
		surf2Dread(&data,  spaceData, 4*idx,0);
		float resultCell = 0.7*data;

		if( (x+1) < X_SIZE )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x+1,y,z),0);
			resultCell +=.03 *data;
		}
		if( (x-1) >= 0 )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x-1,y,z),0);
			resultCell +=.03 *data;
		}
		if( (y+1) < Y_SIZE )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y+1,z),0);
			resultCell +=.03 *data;
		}
		if( (y-1) >= 0 )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y-1,z),0);
			resultCell +=.03 *data;
		}
		if( (z+1) < Z_SIZE )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y,z+1),0);
			resultCell +=.03 *data;
		}
		if( (z-1) >= 0 )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y,z-1),0);
			resultCell +=.03 *data;
		}
		if( (x+2) < X_SIZE )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x+2,y,z),0);
			resultCell +=.02 *data;
		}
		if( (x-2) >= 0 )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x-2,y,z),0);
			resultCell +=.02 *data;
		}
		if( (y+2) < Y_SIZE )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y+2,z),0);
			resultCell +=.02 *data;
		}
		if( (y-2) >= 0 )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y-2,z),0);
			resultCell +=.02 *data;
		}
		if( (z+2) < Z_SIZE )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y,z+2),0);
			resultCell +=.02 *data;
		}
		if( (z-2) >= 0 )
		{
			surf2Dread(&data,  spaceData, 4*IDX_3D(x,y,z-2),0);
			resultCell +=.02 *data;
		}

		surf2Dwrite(resultCell, resultData, 4*idx,0);
	}
}

int blockSizeOf(unsigned size,unsigned thdsInBlock)
{
	return ceil(float(size) / float(thdsInBlock));
}

unsigned int calcSharedSize(dim3 thds)
{
	if(thds.z == 1)
		return sizeof(float) * (thds.x + 4) * (thds.y + 4) ;
	else
		return sizeof(float) * (thds.x + 4) * (thds.y + 4) * (thds.z + 4);
	 // each thread - each cell);
	 // + boundaries threads need neighbours from other block
}

void printOnce(char* text)
{
	static std::once_flag flag;
	std::call_once(flag,[text] { printf(text); });
}

void simulationGlobal(fraction* d_space,fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y,TH_IN_BLCK_Z);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
						  blockSizeOf(Y_SIZE,thds.y),
						  blockSizeOf(Z_SIZE,thds.z));
	printOnce("Global\n");
	stepGlobal<<<numBlocks, thds>>>(d_space,d_result);
}

void simulationSurface(cudaSurfaceObject_t spaceData,cudaSurfaceObject_t resultData)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y,TH_IN_BLCK_Z);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
						  blockSizeOf(Y_SIZE,thds.y),
						  blockSizeOf(Z_SIZE,thds.z));
	printOnce("Surface\n");
	stepSurface<<<numBlocks, thds>>>(spaceData,resultData);
}

void simulationShared3dCube(fraction* d_space,fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y,TH_IN_BLCK_Z);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
						  blockSizeOf(Y_SIZE,thds.y),
						  blockSizeOf(Z_SIZE,thds.z));
	static auto	shMemSize = calcSharedSize(thds);
	printOnce("Shared Cube\n");
	stepShared3DCube<<<numBlocks, thds,shMemSize>>>(d_space,d_result);
}

void simulationShared3dLayer(fraction* d_space,fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
					      blockSizeOf(Y_SIZE,thds.y));
	static auto	shMemSize = calcSharedSize(thds);
	printOnce("Shared Layer by Layer\n");

	for(auto z = 0;z < Z_SIZE; ++z)
		stepShared3DLayer<<<numBlocks, thds,shMemSize>>>(d_space,d_result,z);

}

void simulationShared3dLayerForIn(fraction* d_space,fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
					      blockSizeOf(Y_SIZE,thds.y));
	static auto	shMemSize = calcSharedSize(thds);
	printOnce("Shared Layer by Layer for inside kernel\n");

	stepShared3DLayerForIn<<<numBlocks, thds,shMemSize>>>(d_space,d_result);

}

void simulationShared3dForIn(fraction* d_space,fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X*2, TH_IN_BLCK_Y*2);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
					      blockSizeOf(Y_SIZE,thds.y));
	static auto	shMemSize = calcSharedSize(thds) * Z_SIZE;
	printOnce("Shared For In\n");
	stepSharedForIn<<<numBlocks, thds,shMemSize>>>(d_space,d_result);
}


void simulation(void* space,void* result,enum deviceSimulationType type)
{
	switch(type)
	{
	case GLOBAL:
		simulationGlobal((fraction*)space,(fraction*)result);
		break;
	case SHARED_3D_CUBE:
		simulationShared3dCube((fraction*)space,(fraction*)result);
		break;
	case SHARED_3D_LAYER:
		simulationShared3dLayer((fraction*)space,(fraction*)result);
		break;
	case SHARED_3D_FOR_IN:
		simulationShared3dForIn((fraction*)space,(fraction*)result);
		break;
	case SHARED_3D_LAYER_FOR_IN:
		simulationShared3dLayerForIn((fraction*)space,(fraction*)result);
		break;
	}
    cudaCheckErrors("step failed!");
}



