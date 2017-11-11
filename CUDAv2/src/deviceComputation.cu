#include <helper_cuda.h>
#include <mutex>

#include "Fraction.h"
#include "computation.h"

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

__device__ void fillShMem(int3 idx3,Fraction* space, Fraction shSpace[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4])
{
    const int thx = threadIdx.x + 2, thy = threadIdx.y + 2;
    const int idx = IDX_3D(idx3.x, idx3.y, idx3.z);
    const int x = idx3.x, y = idx3.y, z = idx3.z;
    shSpace[thx][thy] = space[idx];

    if (threadIdx.x == 0 && x > 1)
    {
        shSpace[thx - 2][thy] = space[IDX_3D(x - 2, y, z)];
    }
    if (threadIdx.x == 1 && x > 2)
    {
        shSpace[thx - 2][thy] = space[IDX_3D(x - 2, y, z)];
    }
    if (threadIdx.x == blockDim.x - 2 && x < X_SIZE - 2)
    {
        shSpace[thx + 2][thy] = space[IDX_3D(x + 2, y, z)];
    }
    if (threadIdx.x == blockDim.x - 1 && x < X_SIZE - 1)
    {
        shSpace[thx + 2][thy] = space[IDX_3D(x + 2, y, z)];
    }
    if (threadIdx.y == 0 && y > 1)
    {
        shSpace[thx][thy - 2] = space[IDX_3D(x, y - 2, z)];
    }
    if (threadIdx.y == 1 && y > 2)
    {
        shSpace[thx][thy - 2] = space[IDX_3D(x, y - 2, z)];
    }
    if (threadIdx.y == blockDim.y - 2 && y < Y_SIZE - 2)
    {
        shSpace[thx][thy + 2] = space[IDX_3D(x, y + 2, z)];
    }
    if (threadIdx.y == blockDim.y - 1 && y < Y_SIZE - 1)
    {
        shSpace[thx][thy + 2] = space[IDX_3D(x, y + 2, z)];
    }
}

__global__ void stepShared3DLayer(FluidParams* params, Fraction* spaceData,Fraction* resultData,int z)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Fraction shSpace[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4];
    __syncthreads();// wait for constructors

	if(x<X_SIZE && y<Y_SIZE)
	{
        const int idx = IDX_3D(x, y, z);

        fillShMem(make_int3(x, y, z), spaceData, shSpace);

		__syncthreads(); // wait for threads to fill whole shared memory

		// Calculate cell  with data from shared memory (Layer part x,y)
        resultData[idx] = resultZ(params, spaceData[IDX_3D(x, y, z - 2)],  // zpp
                                          spaceData[IDX_3D(x, y, z - 1)],  // zp
                                          spaceData[idx],                  // cur
                                          spaceData[IDX_3D(x, y, z + 1)],  // zn
                                          spaceData[IDX_3D(x, y, z + 2)],  // znn
                                          shSpace);
	}
}

__global__ void stepShared3DLayerForIn(FluidParams* params, Fraction* spaceData,Fraction* resultData)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Fraction shSpace[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4];
    __syncthreads(); // wait for constructors

	if(x<X_SIZE && y<Y_SIZE)
	{
        for (int z = 2; z < Z_SIZE - 2; ++z)
		{
			const int idx = IDX_3D(x,y,z);
            fillShMem(make_int3(x, y, z), spaceData, shSpace);

            __syncthreads(); // wait for threads to fill whole shared memory

                             // Calculate cell  with data from shared memory (Layer part x,y)
            resultData[idx] = resultZ(params, spaceData[IDX_3D(x, y, z - 2)],  // zpp
                                              spaceData[IDX_3D(x, y, z - 1)],  // zp
                                              spaceData[idx],      // cur
                                              spaceData[IDX_3D(x, y, z + 1)],  // zn
                                              spaceData[IDX_3D(x, y, z + 2)],  // znn
                                              shSpace);

		}
	}
}

__global__ void stepGlobal(FluidParams* params,Fraction* spaceData,Fraction* resultData)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	if(x < X_SIZE - 2 && y < Y_SIZE - 2 && z < Z_SIZE - 2 &&
       x > 1          && y > 1          && z > 1)
	{
        resultData[IDX_3D(x, y, z)] = result3D(params, spaceData, make_int3(x, y, z));
	}
}

__global__ void stepSurface(cudaSurfaceObject_t spaceData,cudaSurfaceObject_t resultData)
{
	/*const int x = blockIdx.x * blockDim.x + threadIdx.x;
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
	}*/
}

int blockSizeOf(unsigned size,unsigned thdsInBlock)
{
	return ceil(float(size) / float(thdsInBlock));
}

void printOnce(char* text)
{
	static std::once_flag flag;
	std::call_once(flag,[text] { printf(text); });
}

void simulationGlobal(FluidParams* d_params,Fraction* d_space,Fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y,TH_IN_BLCK_Z);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
						  blockSizeOf(Y_SIZE,thds.y),
						  blockSizeOf(Z_SIZE,thds.z));
	printOnce("Global\n");
	stepGlobal<<<numBlocks, thds>>>(d_params,d_space,d_result);
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

void simulationShared3dLayer(FluidParams* d_params, Fraction* d_space, Fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
					      blockSizeOf(Y_SIZE,thds.y));
	printOnce("Shared Layer by Layer\n");

    for (int z = 2; z < Z_SIZE - 2; ++z)
		stepShared3DLayer<<<numBlocks, thds>>>(d_params,d_space,d_result,z);

}

void simulationShared3dLayerForIn(FluidParams* d_params, Fraction* d_space, Fraction* d_result)
{
	static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y);
	static dim3 numBlocks(blockSizeOf(X_SIZE,thds.x),
					      blockSizeOf(Y_SIZE,thds.y));
	printOnce("Shared Layer by Layer for inside kernel\n");

	stepShared3DLayerForIn<<<numBlocks, thds>>>(d_params,d_space,d_result);

}

void simulation(void* pars, void* space,void* result,enum deviceSimulationType type)
{
	switch(type)
	{
	case GLOBAL:
		simulationGlobal((FluidParams*)pars,(Fraction*)space,(Fraction*)result);
		break;
	case SHARED_3D_LAYER:
		simulationShared3dLayer((FluidParams*)pars, (Fraction*)space,(Fraction*)result);
		break;
	case SHARED_3D_LAYER_FOR_IN:
		simulationShared3dLayerForIn((FluidParams*)pars, (Fraction*)space,(Fraction*)result);
		break;
	}
    cudaCheckErrors("step failed!");
}



