#include "computation.cuh"

void hostSimulation(fraction * spaceData, fraction *resultData)
{
	int idx;
	float* result = resultData->U;
	float* space  = spaceData->U;

	for(int y=0; y<Y_SIZE;++y)
	{
		for(int x=0; x<X_SIZE;++x)
		{
			idx = IDX_2D(x,y);

			result[idx] = 0.7*space[idx];

			if( (y-1) > 0 )
				result[idx] +=.05 *space[(y-1)*X_SIZE+x];
			if( (y-2) > 0 )
				result[idx] +=.025*space[(y-2)*X_SIZE+x];
			if( (y+1) < Y_SIZE )
				result[idx] +=.05 *space[(y+1)*X_SIZE+x];
			if( (y+2) < Y_SIZE )
				result[idx] +=.025*space[(y+2)*X_SIZE+x];
			if( (x-1) > 0 )
				result[idx] +=.05 *space[(y)*X_SIZE+x-1];
			if( (x-2) > 0 )
				result[idx] +=.025*space[(y)*X_SIZE+x-2];
			if( (x+1) < X_SIZE )
				result[idx] +=.05 *space[(y)*X_SIZE+x+1];
			if( (x+2) < X_SIZE )
				result[idx] +=.025*space[(y)*X_SIZE+x+2];
		}
	}
}
