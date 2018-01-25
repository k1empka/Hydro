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

__device__ void fillShMem(StartArgs args, int3 idx3,Fraction* space, Fraction shSpace[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4])
{
    const int thx = threadIdx.x + 2, thy = threadIdx.y + 2;
    const int idx = args.IDX_3D(idx3.x, idx3.y, idx3.z);
    const int x = idx3.x, y = idx3.y, z = idx3.z;
    shSpace[thx][thy] = space[idx];

    if (threadIdx.x == 0 && x > 1)
    {
        shSpace[thx - 2][thy] = space[args.IDX_3D(x - 2, y, z)];
    }
    if (threadIdx.x == 1 && x > 2)
    {
        shSpace[thx - 2][thy] = space[args.IDX_3D(x - 2, y, z)];
    }
    if (threadIdx.x == blockDim.x - 2 && x < args.X_SIZE - 2)
    {
        shSpace[thx + 2][thy] = space[args.IDX_3D(x + 2, y, z)];
    }
    if (threadIdx.x == blockDim.x - 1 && x < args.X_SIZE - 1)
    {
        shSpace[thx + 2][thy] = space[args.IDX_3D(x + 2, y, z)];
    }
    if (threadIdx.y == 0 && y > 1)
    {
        shSpace[thx][thy - 2] = space[args.IDX_3D(x, y - 2, z)];
    }
    if (threadIdx.y == 1 && y > 2)
    {
        shSpace[thx][thy - 2] = space[args.IDX_3D(x, y - 2, z)];
    }
    if (threadIdx.y == blockDim.y - 2 && y < args.Y_SIZE - 2)
    {
        shSpace[thx][thy + 2] = space[args.IDX_3D(x, y + 2, z)];
    }
    if (threadIdx.y == blockDim.y - 1 && y < args.Y_SIZE - 1)
    {
        shSpace[thx][thy + 2] = space[args.IDX_3D(x, y + 2, z)];
    }
}

__global__ void stepShared3DLayer(StartArgs args, FluidParams* params, Fraction* spaceData,Fraction* resultData,int z)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Fraction shSpace[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4];
    __syncthreads();// wait for constructors

    if(x<args.X_SIZE && y<args.Y_SIZE)
    {
        const int idx = args.IDX_3D(x, y, z);

        fillShMem(args, make_int3(x, y, z), spaceData, shSpace);

        __syncthreads(); // wait for threads to fill whole shared memory

        // Calculate cell  with data from shared memory (Layer part x,y)
        resultData[idx] = resultZ(params, spaceData[args.IDX_3D(x, y, z - 2)],  // zpp
                                          spaceData[args.IDX_3D(x, y, z - 1)],  // zp
                                          spaceData[idx],                  		// cur
                                          spaceData[args.IDX_3D(x, y, z + 1)],  // zn
                                          spaceData[args.IDX_3D(x, y, z + 2)],  // znn
                                          shSpace);
    }
}

__global__ void stepShared3DLayerForIn(StartArgs args, FluidParams* params, Fraction* spaceData,Fraction* resultData)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ Fraction shSpace[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4];
    Fraction zpp, zp, zc, zn, znn;

    if(x<args.X_SIZE && y<args.Y_SIZE)
    {
        for (int z = 2; z < args.Z_SIZE - 2; ++z)
        {
            const int idx = args.IDX_3D(x,y,z);
            fillShMem(args, make_int3(x, y, z), spaceData, shSpace);

            if (z == 2)
            {
                zpp = spaceData[args.IDX_3D(x, y, z - 2)];
                zp = spaceData[args.IDX_3D(x, y, z - 1)];
                zc = spaceData[args.IDX_3D(x, y, z - 1)];
                zn = spaceData[args.IDX_3D(x, y, z + 1)];
            }
            znn = spaceData[args.IDX_3D(x, y, z + 2)];

            __syncthreads(); // wait for threads to fill whole shared memory

                             // Calculate cell  with data from shared memory (Layer part x,y)
            resultData[idx] = resultZ(params, zpp,zp,zc,zn,znn,shSpace);
            zpp = zp;
            zp = zc;
            zc = zn;
            zn = znn;
        }
    }
}

__global__ void stepGlobal(StartArgs args, FluidParams* params,Fraction* spaceData,Fraction* resultData)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x < args.X_SIZE - 2 && y < args.Y_SIZE - 2 && z < args.Z_SIZE - 2 &&
       x > 1          && y > 1          && z > 1)
    {
        resultData[args.IDX_3D(x, y, z)] = result3D(args, params, spaceData, make_int3(x, y, z));
    }
}

__device__ void writeSurface(Fraction f,cudaSurfaceObject_t data,const int x,const int y,const int z)
{

	static const int SIZE_OF_FLOAT = sizeof(float);

	surf3Dwrite((f.E), data, SIZE_OF_FLOAT*(5*x),  y, z);
	surf3Dwrite((f.R), data, SIZE_OF_FLOAT*(5*x+1),y, z);
	surf3Dwrite((f.Vx),data, SIZE_OF_FLOAT*(5*x+2),y, z);
	surf3Dwrite((f.Vy),data, SIZE_OF_FLOAT*(5*x+3),y, z);
	surf3Dwrite((f.Vz),data, SIZE_OF_FLOAT*(5*x+4),y, z);

}

__global__ void stepSurface(StartArgs args, FluidParams* params,cudaSurfaceObject_t spaceData,cudaSurfaceObject_t resultData)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < args.X_SIZE - 2 && y < args.Y_SIZE - 2 && z < args.Z_SIZE - 2 &&
        x > 1 && y > 1 && z > 1) 
    {
        //const int idx = IDX_3D(x,y,z);
        const Fraction output = result3DSurface(params,spaceData,make_int3(x, y, z));

        writeSurface(output,resultData,x,y,z);
    }
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

void simulationGlobal(StartArgs args, FluidParams* d_params,Fraction* d_space,Fraction* d_result)
{
    static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y,TH_IN_BLCK_Z);
    static dim3 numBlocks(blockSizeOf(args.X_SIZE,thds.x),
                          blockSizeOf(args.Y_SIZE,thds.y),
                          blockSizeOf(args.Z_SIZE,thds.z));
    printOnce("Global\n");
    stepGlobal<<<numBlocks, thds>>>(args,d_params,d_space,d_result);
}

void simulationSurface(StartArgs args, FluidParams* params,cudaSurfaceObject_t spaceData,cudaSurfaceObject_t resultData)
{
    static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y,TH_IN_BLCK_Z);
    static dim3 numBlocks(blockSizeOf(args.X_SIZE,thds.x),
                          blockSizeOf(args.Y_SIZE,thds.y),
                          blockSizeOf(args.Z_SIZE,thds.z));
    printOnce("Surface\n");
    stepSurface<<<numBlocks, thds>>>(args, params,spaceData,resultData);
}

void simulationShared3dLayer(StartArgs args, FluidParams* d_params, Fraction* d_space, Fraction* d_result)
{
    static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y);
    static dim3 numBlocks(blockSizeOf(args.X_SIZE,thds.x),
                          blockSizeOf(args.Y_SIZE,thds.y));
    printOnce("Shared Layer by Layer\n");

    for (int z = 2; z < args.Z_SIZE - 2; ++z)
        stepShared3DLayer<<<numBlocks, thds>>>(args,d_params,d_space,d_result,z);

}

void simulationShared3dLayerForIn(StartArgs args, FluidParams* d_params, Fraction* d_space, Fraction* d_result)
{
    static dim3 thds(TH_IN_BLCK_X, TH_IN_BLCK_Y);
    static dim3 numBlocks(blockSizeOf(args.X_SIZE,thds.x),
                          blockSizeOf(args.Y_SIZE,thds.y));
    printOnce("Shared Layer by Layer for inside kernel\n");

    stepShared3DLayerForIn<<<numBlocks, thds>>>(args,d_params,d_space,d_result);

}

void simulation(StartArgs args, FluidParams* pars, void* space,void* result)
{
    switch(args.type)
    {
    case deviceSimulationType::GLOBAL:
        simulationGlobal(args,pars,(Fraction*)space,(Fraction*)result);
        break;
    case deviceSimulationType::SHARED_3D_LAYER:
        simulationShared3dLayer(args,pars, (Fraction*)space,(Fraction*)result);
        break;
    case deviceSimulationType::SHARED_3D_LAYER_FOR_IN:
        simulationShared3dLayerForIn(args,pars, (Fraction*)space,(Fraction*)result);
        break;
    }
    cudaCheckErrors("step failed!");
}
