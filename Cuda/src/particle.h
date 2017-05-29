/*
 * cell.h
 *
 *  Created on: 20 kwi 2017
 *      Author: mknap
 */

#ifndef CELL_H_
#define CELL_H_

#include "vector_types.h"

// Factors needed in fluids computation
namespace Factors
{
	const float T = 0.5f; // Time period between iterations measured e.g. in seconds
	const float VISCOSITY = 0.02f; // just guess for now
	const int	 X_SIZE = 100;
	const int	 Y_SIZE = 100;
	const int	 Z_SIZE = 100;
	const int 	 INIT_FLUID_W = 10;
	const int    TOTAL_SIZE = X_SIZE * Y_SIZE * Z_SIZE;
	const int	 P_COUNT = 1000000; // Number of particles
	const int	 FORCE  =  (5.8f*1000) ; // External force to make simulation interesting
	const int    FORCE_RADIUS = 4;
	const int 	 ITERATION_NUM = 100;
	const float  GRAVITY = 9.80665;
	const float  MASS_P = 20;
}

namespace CudaParams
{
	const int BLOCK_SIZE = 128;
	const int RANGE_TH = 16; //Range of calculation for one thread in kernel
	const int P_NUM_PER_BUCKET = 20;
}


struct Particle
{
	float4  position[Factors::P_COUNT]; //float4 to have equal bytes, increase performance
	float4  velocity[Factors::P_COUNT];
	float4  velocityTmp[Factors::P_COUNT]; // Temporary storage is needed, operations cannot be performed in place
	float4	externalForces[Factors::P_COUNT];
	int     index[Factors::TOTAL_SIZE]; // map to grid

	//TO DO
	//float2  externalForce;
	//float  density; // density
	//float  pressure; // pressure
};

#endif /* CELL_H_ */
