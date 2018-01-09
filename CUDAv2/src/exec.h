#pragma once

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "utils.h"
#include "customTimer.h"
#include "printer.h"
#include "Fraction.h"

#define RANDOM false

Fraction* execDeviceSurface(StartArgs args, FluidParams* params, Fraction* space);
Fraction* execDevice(StartArgs args);
Fraction* execHost(StartArgs args);
Fraction* execHostOctree(StartArgs args);

