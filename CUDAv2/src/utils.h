#pragma once

#include "computation.h"
#include <stdio.h>

void 	  initCuda();
void  	  swapPointers(void*& p1,void*& p2);
fraction* initSpace(const bool random);
void      compare_results(fraction* space1,fraction* space2);
void      printData(float* data);
