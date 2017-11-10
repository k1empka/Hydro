#pragma once

#include "computation.h"
#include <stdio.h>

void 	    initCuda();
Fraction*   initSpace(const bool random);
FluidParams initParams();
void  	    swapPointers(void*& p1, void*& p2);
void        compare_results(Fraction* hostSpace,Fraction* deviceSpace);
void        printData(float* data);
