#pragma once

#define NUM_OF_ITERATIONS 100
#define X_SIZE 100
#define Y_SIZE 100
#define Z_SIZE 25//FORINVERSION USES A LOTS OF SHERED MEMORY DEPENDED ON Z_SIZE
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
	float  Vx[X_SIZE*Y_SIZE*Z_SIZE];
	float  Vy[X_SIZE*Y_SIZE*Z_SIZE];
	//TODO more paramas
};

void simulation(fraction* space,fraction* result);
void hostSimulation(fraction* space,fraction* result);
