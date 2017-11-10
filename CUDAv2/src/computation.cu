#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "computation.h"
#include "Fraction.h"

#define REAL float

__host__ __device__ REAL slopeLimiter(REAL r)
{
    REAL rr = r;
    return (rr < 1) ? (rr < 0) ? 0 : rr : 1;
}

__host__ __device__ Fraction slopeLimiterFraction(Fraction r)
{
    float E = r.E;
    float R = r.R;
    float Mx = r.Vx;
    float My = r.Vy;
    float Mz = r.Vz;
    return Fraction(slopeLimiter(E), slopeLimiter(R),
        make_float3(slopeLimiter(Mx), slopeLimiter(My), slopeLimiter(Mz)));
}

// MUSCL algorithm -- calc initial conditions for Musta-Force with slope limiting, in given direction
__host__ __device__ Fraction2 muscl(eDim dim, FluidParams pars, Fraction Up, Fraction Uc, Fraction Un)
{
    Fraction2 result;

    Fraction r = (Uc - Up) / (Un - Uc);
    Fraction xi = slopeLimiterFraction(r);
    Fraction D = ((Uc - Up) * 0.5 * (1 + pars.omega) + (Un - Uc) * 0.5 * (1 - pars.omega)) * xi;

    Fraction UL = Uc - D * 0.5;
    Fraction UR = Uc + D * 0.5;

    REAL dx;

    switch (dim)
    {
    case eDim::x:
        dx = pars.d.x;
        break;

    case eDim::y:
        dx = pars.d.y;
        break;

    case eDim::z:
        dx = pars.d.z;
        break;
    }

    Fraction adv = (UL.flux(dim) - UR.flux(dim)) * (pars.d.w / dx * 0.5);

    result.L = UL - adv;
    result.R = UR - adv;

    return result;
}

// Musta-Force in given direction
__host__ __device__ Fraction mustaForce(eDim dim, FluidParams pars, Fraction L0, Fraction R0)
{
    Fraction result;
    Fraction UL = L0, UR = R0;
    int i = 0;

    REAL dx;

    switch (dim)
    {
    case eDim::x:
        dx = pars.d.x;
        break;

    case eDim::y:
        dx = pars.d.y;
        break;

    case eDim::z:
        dx = pars.d.z;
        break;
    }

    REAL alpha = pars.d.w / dx;

    do
    {
        Fraction FL = UL.flux(dim);
        Fraction FR = UR.flux(dim);

        Fraction UM = (UL + UR) * 0.5 - (FL - FR) * (0.5 * alpha);
        Fraction FM = UM.flux(dim);

        result = (FL + FM * 2 + FR - (UR - UL) / alpha) * 0.25;

        if (i == pars.mustaSteps - 1)
            break;

        UL = UL - (result - FL) * alpha;
        UR = UR - (FR - result) * alpha;

        i++;
    } while (true);

    return result;
}

__host__ __device__ Fraction fluidAlgorithm(eDim dim, FluidParams pars, Fraction pp, Fraction p,
                                                               Fraction c,  Fraction n, Fraction nn)
{
    REAL dx;

    switch (dim)
    {
    case eDim::x:
        dx = pars.d.x;
        break;

    case eDim::y:
        dx = pars.d.y;
        break;

    case eDim::z:
        dx = pars.d.z;
        break;
    }

    REAL alpha = pars.d.w / dx;
    Fraction2 LRp, LR, LRn;

    LR = muscl(dim, pars, p, c, n);
    LRp = muscl(dim, pars, pp, p, c);
    LRn = muscl(dim, pars, c, n, nn);

    LR.R = mustaForce(dim, pars, LR.R, LRn.L);
    LR.L = mustaForce(dim, pars, LRp.R, LR.L);

    return (LR.L - LR.R) * alpha;
}

__host__ __device__ Fraction
result3D(FluidParams pars, Fraction* data,int3 pos)
{
    Fraction cur;
    Fraction result = cur = data[IDX_3D(pos.x, pos.y, pos.z)];

    {
        Fraction xpp = data[IDX_3D(pos.x - 2, pos.y, pos.z)],
            xp = data[IDX_3D(pos.x - 1, pos.y, pos.z)],
            xn = data[IDX_3D(pos.x + 1, pos.y, pos.z)],
            xnn = data[IDX_3D(pos.x + 2, pos.y, pos.z)];
        result = result + fluidAlgorithm(eDim::x, pars, xpp, xp, cur, xn, xnn);
    }
    {
        Fraction ypp = data[IDX_3D(pos.x, pos.y - 2, pos.z)],
            yp = data[IDX_3D(pos.x, pos.y - 1, pos.z)],
            yn = data[IDX_3D(pos.x, pos.y + 1, pos.z)],
            ynn = data[IDX_3D(pos.x, pos.y + 2, pos.z)];
        result = result + fluidAlgorithm(eDim::y, pars, ypp, yp, cur, yn, ynn);
    }
    {
        Fraction zpp = data[IDX_3D(pos.x, pos.y, pos.z - 2)],
            zp = data[IDX_3D(pos.x , pos.y, pos.z - 1)],
            zn = data[IDX_3D(pos.x, pos.y, pos.z + 1)],
            znn = data[IDX_3D(pos.x, pos.y, pos.z + 2)];
        result = result + fluidAlgorithm(eDim::z, pars, zpp, zp, cur, zn, znn);
    }
    return result;
}
__device__ Fraction 
resultZ(FluidParams pars,Fraction zpp, Fraction zp, Fraction cur, Fraction zn,
        Fraction znn, Fraction storage[TH_IN_BLCK_Y + 4][TH_IN_BLCK_X + 4])
{
    Fraction result = cur;

    {
        // update in Z dimension
        result = result + fluidAlgorithm(eDim::z, pars, zpp, zp, cur, zn, znn);

        Fraction xpp = storage[threadIdx.x - 2][threadIdx.y],
            xp = storage[threadIdx.x - 1][threadIdx.y],
            xn = storage[threadIdx.x + 1][threadIdx.y],
            xnn = storage[threadIdx.x + 2][threadIdx.y];

        result = result + fluidAlgorithm(eDim::x, pars, xpp, xp, cur, xn, xnn);

        // update in Y dimension
        // fetch Y neighbours from shared memory
        Fraction ypp = storage[threadIdx.x][threadIdx.y - 2],
            yp = storage[threadIdx.x][threadIdx.y - 1],
            yn = storage[threadIdx.x][threadIdx.y + 1],
            ynn = storage[threadIdx.x][threadIdx.y + 2];

        result = result + fluidAlgorithm(eDim::y, pars, ypp, yp, cur, yn, ynn);
    }

    return Fraction(result.E, result.R, make_float3(result.Vx, result.Vy, result.Vz));
}
