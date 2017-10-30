#pragma once

#include "computation.h"
#include <stdio.h>

void 	  initCuda();
void  	  swapPointers(void*& p1,void*& p2);
fraction* initSpace(const bool random);
void      compare_results(fraction* hostSpace,fraction* deviceSpace);
void      printData(float* data);
