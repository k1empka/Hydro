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
	const int	 FORCE  =  (5.8f*1000) ; // External force to make simulation interesting
	const int    FORCE_RADIUS = 4;
	const int 	 ITERATION_NUM = 100;
}
namespace CudaParams
{
	const int BLCK_X = 8;
	const int BLCK_Y = 8;
	const int BLCK_Z = 8;
	const int RANGE_TH = 16; //Range of calculation for one thread in kernel
}


struct Particle
{
	float4  position[Factors::TOTAL_SIZE]; //float4 to have equal bytes, increase performance
	float4  velocity[Factors::TOTAL_SIZE];
	float4  velocityTmp[Factors::TOTAL_SIZE]; // Temporary storage is needed, operations cannot be performed in place

	float2  externalForce;


	//TO DO
	//float  density; // density
	//float  pressure; // pressure
};

#endif /* CELL_H_ */
