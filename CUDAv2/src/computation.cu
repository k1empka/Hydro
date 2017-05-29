#include "computation.cuh"

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


void simulation(fraction* d_space,fraction* d_result)
{
	static int N = X_SIZE * Y_SIZE;
	static dim3 threadsPerBlock(16, 16);
	static dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	step<<<numBlocks, threadsPerBlock>>>(d_space,d_result);
	copySpace<<<numBlocks, threadsPerBlock>>>(d_result,d_space);
}
