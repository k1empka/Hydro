#pragma once

#include "computation.h"
#include <stdio.h>

void 	    initCuda();
Fraction*   initSpace(StartArgs args);
FluidParams initParams();
void  	    swapPointers(void*& p1, void*& p2);
void        compare_results(StartArgs args,Fraction* hostSpace,Fraction* deviceSpace);
float* 		spaceToFloats(StartArgs args,Fraction* space);
void 		floatsToSpace(StartArgs args,float* floats,Fraction* space);
