#pragma once

#define NUM_OF_ITERATIONS 100
#define X_SIZE 100
#define Y_SIZE 100
#define NUM_OF_START_FRACTIONS 100
#define MAX_START_FORCE 100
#define IDX_2D(x,y) ((y) * X_SIZE + x)
#define DT 0.5
#define uV 0.02 // viscosity
#define H  1 // radius for SPH
#define P_MASS 2 // mass - no idea about value

#define TH_IN_BLCK_X 16
#define TH_IN_BLCK_Y 16
#define THX_2D(x,y) ((y) * TH_IN_BLCK_X + x)
/*
 *
 * SPH - smoothing particle hydrodynamics
 *  more info: http://matthias-mueller-fischer.ch/publications/sca03.pdf
 */

struct fraction
{
	float  U[X_SIZE*Y_SIZE];
	float  Vx[X_SIZE*Y_SIZE];
	float  Vy[X_SIZE*Y_SIZE];
	//TODO more paramas
};

void simulation(fraction* space,fraction* result);
