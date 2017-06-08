#include "computation.cuh"

void hostSimulation(fraction * spaceData, fraction *resultData)
{
	int idx;
	float* result = resultData->U;
	float* space  = spaceData->U;

	for(int z=0; z<Z_SIZE;++z)
	{
		for(int y=0; y<Y_SIZE;++y)
		{
			for(int x=0; x<X_SIZE;++x)
			{
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
	}
}
