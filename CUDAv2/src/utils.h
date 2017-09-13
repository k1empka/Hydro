#pragma once

#include "computation.cuh"
#include <stdio.h>


void 	  initCuda();
void	  printHeader(FILE* f);
void      printIteration(FILE* f,fraction* space, int iter);
void  	  swapFractionPointers(fraction*& p1,fraction*& p2);
FILE*	  initOutputFile(bool hostSimulation);
fraction* initSpace(const bool random);
void      compare_results(fraction* space1,fraction* space2);
