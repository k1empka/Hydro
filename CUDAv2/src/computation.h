#pragma once

#include <cuda_runtime.h>

#define NUM_OF_ITERATIONS 100
#define X_SIZE 10
#define Y_SIZE 10
#define Z_SIZE 10
#define NUM_OF_START_FRACTIONS 100
#define MAX_START_FORCE 98
#define DT 0.5
#define uV 0.02 // viscosity
#define H  1 // radius for SPH
#define P_MASS 2 // mass - no idea about value

#define TH_IN_BLCK_X 8
#define TH_IN_BLCK_Y 8
#define TH_IN_BLCK_Z 8
#define NUM_NEIGH 4 // neighbours number affected for single cell in row
#define THX_2D(x,y) ((y) * (TH_IN_BLCK_X+NUM_NEIGH) + x)
#define THX_3D(x,y,z) ((z) * ((TH_IN_BLCK_Y+NUM_NEIGH) * (TH_IN_BLCK_X+NUM_NEIGH)) + (y) * (TH_IN_BLCK_X+NUM_NEIGH) + x)
#define IDX_2D(x,y) ((y) * X_SIZE + x)
#define IDX_3D(x,y,z) ((z) * (Y_SIZE * X_SIZE) +(y )* X_SIZE + x)

/*
 *
 * SPH - smoothing particle hydrodynamics
 *  more info: http://matthias-mueller-fischer.ch/publications/sca03.pdf
 */

struct fraction
{
	float  U[X_SIZE*Y_SIZE*Z_SIZE];
	float  Vx[X_SIZE];
	float  Vy[Y_SIZE];
	//TODO more paramas
};

enum deviceSimulationType{GLOBAL,SURFACE,SHARED_3D_CUBE,SHARED_3D_LAYER,SHARED_3D_FOR_IN,SHARED_3D_LAYER_FOR_IN};

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

void simulation(void* space,void* result,enum deviceSimulationType type);
void hostSimulation(void* space,void* result);
void simulationSurface(cudaSurfaceObject_t spaceData,cudaSurfaceObject_t resultData);
