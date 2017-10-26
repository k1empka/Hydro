#pragma once

#include "computation.h"
#include <stdio.h>

void 	  initCuda();
void  	  swapFractionPointers(fraction*& p1,fraction*& p2);
fraction* initSpace(const bool random);
void      compare_results(fraction* space1,fraction* space2);
void      printData(float* data);