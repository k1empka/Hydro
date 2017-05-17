#ifndef COMPUTATION_H_
#define COMPUTATION_H_

#include "particle.h"
#include <iostream>
#include <cuda_runtime.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while(0);

int  IDX_3D(int x,int y,int z);
void simulateFluids(Particle* data);


#endif
