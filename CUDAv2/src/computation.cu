#include "computation.cuh"

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
	}
}

__global__ void step(fraction* spaceData,fraction* resultData)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x > 1 && y > 1 && x<X_SIZE && y<Y_SIZE)
	{
		float* result = resultData->U;
		float* space  = spaceData->U;

		result[y*X_SIZE+x]=.7*space[y*X_SIZE+x];

		if( (y-1) > 0 )
			result[y*X_SIZE+x] +=.05 *space[(y-1)*X_SIZE+x];
		if( (y-2) > 0 )
			result[y*X_SIZE+x] +=.025*space[(y-2)*X_SIZE+x];
		if( (y+1) < Y_SIZE )
			result[y*X_SIZE+x] +=.05 *space[(y+1)*X_SIZE+x];
		if( (y+2) < Y_SIZE )
			result[y*X_SIZE+x] +=.025*space[(y+2)*X_SIZE+x];
		if( (x-1) > 0 )
			result[y*X_SIZE+x] +=.05 *space[(y)*X_SIZE+x-1];
		if( (x-2) > 0 )
			result[y*X_SIZE+x] +=.025*space[(y)*X_SIZE+x-2];
		if( (x+1) < X_SIZE )
			result[y*X_SIZE+x] +=.05 *space[(y)*X_SIZE+x+1];
		if( (x+2) < X_SIZE )
			result[y*X_SIZE+x] +=.025*space[(y)*X_SIZE+x+2];
	}
}

__global__ void copySpace(fraction* from,fraction* to)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x<X_SIZE && y<Y_SIZE)
	{
		to->U[y*X_SIZE+x]=from->U[y*X_SIZE+x];
	}
}


void simulation(fraction* d_space,fraction* d_result)
{
	static int N = X_SIZE * Y_SIZE;
	static dim3 threadsPerBlock(16, 16);
	static dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	step<<<numBlocks, threadsPerBlock>>>(d_space,d_result);
	copySpace<<<numBlocks, threadsPerBlock>>>(d_result,d_space);
}
