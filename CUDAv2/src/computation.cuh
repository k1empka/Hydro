#pragma once

#define NUM_OF_ITERATIONS 100
#define X_SIZE 100
#define Y_SIZE 100
#define NUM_OF_START_FRACTIONS 100
#define MAX_START_FORCE 100
#define IDX_2D(x,y) (y * X_SIZE + x)

struct fraction
{
	float U;
	//TODO more paramas
};

void simulation(fraction* space,fraction* result);
