#pragma once
#include <cuda_runtime.h>

#include "Fraction.h"

#define NUM_OF_ITERATIONS 10
#define X_SIZE 10
#define Y_SIZE 10
#define Z_SIZE 10
#define SIZE (X_SIZE * Y_SIZE * Z_SIZE)
#define NUM_OF_START_FRACTIONS 100
#define MAX_START_FORCE 100
#define MAX_VELOCITY 10

  
#define TH_IN_BLCK_X 8
#define TH_IN_BLCK_Y 8
#define TH_IN_BLCK_Z 4
#define NUM_NEIGH 4 // neighbours number affected for single cell in row
#define THX_2D(x,y) ((y) * (TH_IN_BLCK_X+NUM_NEIGH) + x)
#define THX_3D(x,y,z) ((z) * ((TH_IN_BLCK_Y+NUM_NEIGH) * (TH_IN_BLCK_X+NUM_NEIGH)) + (y) * (TH_IN_BLCK_X+NUM_NEIGH) + x)
#define IDX_2D(x,y) ((y) * X_SIZE + x)
#define IDX_3D(x,y,z) ((z) * (Y_SIZE * X_SIZE) + y * X_SIZE + x)

/*
 *
 * SPH - smoothing particle hydrodynamics
 *  more info: http://matthias-mueller-fischer.ch/publications/sca03.pdf
 */

enum deviceSimulationType{GLOBAL,SURFACE,SHARED_3D_LAYER,SHARED_3D_LAYER_FOR_IN};

struct FluidParams
{
    float4 d;
    float  omega;
    float  mustaSteps;
};

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

void simulation(FluidParams* params, void* space,void* result,enum deviceSimulationType type);
void hostSimulation(FluidParams* params, void* space,void* result);
void simulationSurface(FluidParams* params,cudaSurfaceObject_t spaceData,cudaSurfaceObject_t resultData);


__host__ __device__ Fraction result3D(FluidParams* pars, Fraction* data, int3 pos);
__device__ Fraction result3DSurface(FluidParams* pars, cudaSurfaceObject_t data, int3 pos);
__device__ Fraction resultZ(FluidParams* pars, Fraction zpp, Fraction zp, Fraction cur, Fraction zn,
                            Fraction znn, Fraction storage[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4]);
