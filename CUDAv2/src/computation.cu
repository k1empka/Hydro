#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "computation.h"
#include "Fraction.h"

#define REAL float

__host__ __device__ Fraction Fraction::flux(eDim dim)
{
    Fraction result;
    float3 v, FM;
    float3 M = make_float3(Vx, Vy, Vz);
    float Mw = sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
    vp vnep = vnep_calc();   ///calculate vnep

#define SMALL(x) (fabs(x)<1e-6)

    switch (dim)
    {
    case eDim::x:
        v.x = (SMALL(Mw)) ? 0 : Vx / Mw * vnep.v;
        FM = make_float3(M.x* v.x, M.y* v.x, M.z* v.x);;
        FM.x += vnep.p;
        result = Fraction(v.x * (E + vnep.p), R * v.x, FM);
        break;

    case eDim::y:
        v.y = (SMALL(Mw)) ? 0 : Vz / Mw * vnep.v;
        FM = make_float3(M.x* v.y, M.y* v.y, M.z* v.y);;
        FM.y += vnep.p;
        result = Fraction(v.y * (E + vnep.p), R * v.y, FM);
        break;

    case eDim::z:
        v.z = (SMALL(Mw)) ? 0 : Vz / Mw * vnep.v;
        FM = make_float3(M.x* v.z, M.y* v.z, M.z* v.z);
        FM.z += vnep.p;
        result = Fraction(v.z * (E + vnep.p), R * v.z, FM);
        break;
    }

#undef SMALL

    return result;
}

__host__ __device__ float eos(float e, float n)
{
    return 0.0; //0.3333333 * e;
}

__host__ __device__ vp Fraction::vnep_calc()
{
    float v = 0;
    float M = sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
    float e = E, n = R, p;
    float step, vdiv, delta, pr_delta, eps = 1e-6;

#define SMALL(x) (fabs(x)<eps)

    if (SMALL(M))
    {
        p = eos(E, R);
    }
    else
    {
        //	if (SMALL(E-M))
        if (SMALL(E - M) || (E < M))
        {
            v = 1.0;
            p = 0.0;
        }
        else
        {
            step = 1.0;
            v = 0.0;
            vdiv = E + eos(E, R);

            vdiv = M / vdiv; // FIXME: potential division by 0
            pr_delta = vdiv - v;

            if (!SMALL(fabs(pr_delta)))
            {
                v += step;

                e = E - M * v;
                float x = 1 - v * v;
                n = R * sqrt((x > 0) ? x : 0);
                p = eos(e, n);

                vdiv = E + p;
                vdiv = M / vdiv; // FIXME: potential division by 0

                delta = vdiv - v;

                if ((signbit(delta) == signbit(pr_delta)) || (fabs(delta) >= eps))
                {
                    delta = eps * signbit(delta);

                    while ((v <= 1.0) && (fabs(delta) >= eps))
                    {
                        while ((v <= 1.0) && (signbit(delta) == signbit(pr_delta)))
                        {
                            pr_delta = delta;
                            v += step;

                            e = E - M * v;
                            float x = 1 - v * v;
                            n = R * sqrt((x > 0) ? x : 0);

                            p = eos(e, n);

                            vdiv = E + p;
                            vdiv = M / vdiv; // FIXME: potential division by 0
                            delta = vdiv - v;
                        }

                        step = -step / 2;
                        pr_delta = delta;
                    }
                }
            }
        }
    }

    e = E - M * v;
    float x = 1 - v * v;
    n = R * sqrt((x > 0) ? x : 0);

    p = eos(e, n);

#undef SMALL
    vp res;
    res.v = v;
    res.p = p;
    return res;
}

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
__host__ __device__ Fraction2 muscl(eDim dim, FluidParams* pars, Fraction Up, Fraction Uc, Fraction Un)
{
    Fraction2 result;

    Fraction r = (Uc - Up) / (Un - Uc);
    Fraction xi = slopeLimiterFraction(r);
    Fraction D = ((Uc - Up) * 0.5 * (1 + pars->omega) + (Un - Uc) * 0.5 * (1 - pars->omega)) * xi;

    Fraction UL = Uc - D * 0.5;
    Fraction UR = Uc + D * 0.5;

    REAL dx;

    switch (dim)
    {
    case eDim::x:
        dx = pars->d.x;
        break;

    case eDim::y:
        dx = pars->d.y;
        break;

    case eDim::z:
        dx = pars->d.z;
        break;
    }

    Fraction adv = (UL.flux(dim) - UR.flux(dim)) * (pars->d.w / dx * 0.5);

    result.L = UL - adv;
    result.R = UR - adv;

    return result;
}

// Musta-Force in given direction
__host__ __device__ Fraction mustaForce(eDim dim, FluidParams* pars, Fraction L0, Fraction R0)
{
    Fraction result;
    Fraction UL = L0, UR = R0;
    int i = 0;

    REAL dx;

    switch (dim)
    {
    case eDim::x:
        dx = pars->d.x;
        break;

    case eDim::y:
        dx = pars->d.y;
        break;

    case eDim::z:
        dx = pars->d.z;
        break;
    }

    REAL alpha = pars->d.w / dx;

    do
    {
        Fraction FL = UL.flux(dim);
        Fraction FR = UR.flux(dim);

        Fraction UM = (UL + UR) * 0.5 - (FL - FR) * (0.5 * alpha);
        Fraction FM = UM.flux(dim);

        result = (FL + FM * 2 + FR - (UR - UL) / alpha) * 0.25;

        if (i == pars->mustaSteps - 1)
            break;

        UL = UL - (result - FL) * alpha;
        UR = UR - (FR - result) * alpha;

        i++;
    } while (true);

    return result;
}

__host__ __device__ Fraction fluidAlgorithm(eDim dim, FluidParams* pars, Fraction pp, Fraction p,
                                                             Fraction c,  Fraction n, Fraction nn)
{
    REAL dx;

    switch (dim)
    {
    case eDim::x:
        dx = pars->d.x;
        break;

    case eDim::y:
        dx = pars->d.y;
        break;

    case eDim::z:
        dx = pars->d.z;
        break;
    }

    REAL alpha = pars->d.w / dx;
    Fraction2 LRp, LR, LRn;

    LR = muscl(dim, pars, p, c, n);
    LRp = muscl(dim, pars, pp, p, c);
    LRn = muscl(dim, pars, c, n, nn);

    LR.R = mustaForce(dim, pars, LR.R, LRn.L);
    LR.L = mustaForce(dim, pars, LRp.R, LR.L);

    return (LR.L - LR.R) * alpha;
}

__host__ __device__ Fraction result3D(StartArgs args, FluidParams* pars, Fraction* data,int3 pos)
{
    Fraction cur;
    Fraction result = cur = data[args.IDX_3D(pos.x, pos.y, pos.z)];
    {
        Fraction xpp = data[args.IDX_3D(pos.x - 2, pos.y, pos.z)],
            xp = data[args.IDX_3D(pos.x - 1, pos.y, pos.z)],
            xn = data[args.IDX_3D(pos.x + 1, pos.y, pos.z)],
            xnn = data[args.IDX_3D(pos.x + 2, pos.y, pos.z)];
        result = result + fluidAlgorithm(eDim::x, pars, xpp, xp, cur, xn, xnn);
    }
    {
        Fraction ypp = data[args.IDX_3D(pos.x, pos.y - 2, pos.z)],
            yp = data[args.IDX_3D(pos.x, pos.y - 1, pos.z)],
            yn = data[args.IDX_3D(pos.x, pos.y + 1, pos.z)],
            ynn = data[args.IDX_3D(pos.x, pos.y + 2, pos.z)];
        result = result + fluidAlgorithm(eDim::y, pars, ypp, yp, cur, yn, ynn);
    }
    {
        Fraction zpp = data[args.IDX_3D(pos.x, pos.y, pos.z - 2)],
            zp = data[args.IDX_3D(pos.x , pos.y, pos.z - 1)],
            zn = data[args.IDX_3D(pos.x, pos.y, pos.z + 1)],
            znn = data[args.IDX_3D(pos.x, pos.y, pos.z + 2)];
        result = result + fluidAlgorithm(eDim::z, pars, zpp, zp, cur, zn, znn);
    }
    return result;
}

__device__ Fraction readFraction(cudaSurfaceObject_t data,const int x,const int y,const int z)
{

	static const int SIZE_OF_FLOAT = sizeof(float);

	Fraction f;

	surf3Dread(&(f.E), data, SIZE_OF_FLOAT*(5*x),  y, z);
	surf3Dread(&(f.R), data, SIZE_OF_FLOAT*(5*x+1),y, z);
	surf3Dread(&(f.Vx),data, SIZE_OF_FLOAT*(5*x+2),y, z);
	surf3Dread(&(f.Vy),data, SIZE_OF_FLOAT*(5*x+3),y, z);
	surf3Dread(&(f.Vz),data, SIZE_OF_FLOAT*(5*x+4),y, z);

	return f;
}

__device__ Fraction result3DSurface(FluidParams* pars, cudaSurfaceObject_t data, int3 pos)
{
    Fraction cur, result;

    cur = readFraction(data,pos.x, pos.y, pos.z);

    result = cur;

    {
        Fraction xpp,xp,xn,xnn;
		xpp=readFraction(data,pos.x - 2, pos.y, pos.z);
		xp =readFraction(data,pos.x - 1, pos.y, pos.z);
		xn =readFraction(data,pos.x + 1, pos.y, pos.z);
		xnn=readFraction(data,pos.x + 2, pos.y, pos.z);
        result = result + fluidAlgorithm(eDim::x, pars, xpp, xp, cur, xn, xnn);
    }
    {
        Fraction ypp,yp,yn,ynn;
        ypp=readFraction(data,pos.x, pos.y - 2, pos.z);
		yp =readFraction(data,pos.x, pos.y - 1, pos.z);
		yn =readFraction(data,pos.x, pos.y + 1, pos.z);
		ynn=readFraction(data,pos.x, pos.y + 2, pos.z);
        result = result + fluidAlgorithm(eDim::y, pars, ypp, yp, cur, yn, ynn);
    }
    {
        Fraction zpp,zp,zn,znn;
        zpp=readFraction(data,pos.x, pos.y, pos.z - 2);
		zp =readFraction(data,pos.x, pos.y, pos.z - 1);
		zn =readFraction(data,pos.x, pos.y, pos.z + 1);
		znn=readFraction(data,pos.x, pos.y, pos.z + 2);
        result = result + fluidAlgorithm(eDim::z, pars, zpp, zp, cur, zn, znn);
    }
    return result;
}

__device__ Fraction resultZ(FluidParams* pars, Fraction zpp, Fraction zp, Fraction cur, Fraction zn,
                            Fraction znn, Fraction storage[TH_IN_BLCK_X + 4][TH_IN_BLCK_Y + 4])
{
    Fraction result = cur;

    // update in Z dimension
    result = result + fluidAlgorithm(eDim::z, pars, zpp, zp, cur, zn, znn);

    int idx = threadIdx.x + 2, idy = threadIdx.y + 2;
    {
        Fraction xpp = storage[idx - 2][idy],
            xp = storage[idx - 1][idy],
            xn = storage[idx + 1][idy],
            xnn = storage[idx + 2][idy];

        result = result + fluidAlgorithm(eDim::x, pars, xpp, xp, cur, xn, xnn);
    }
    {
        // update in Y dimension
        // fetch Y neighbours from shared memory
        Fraction ypp = storage[idx][idy - 2],
            yp = storage[idx][idy - 1],
            yn = storage[idx][idy + 1],
            ynn = storage[idx][idy + 2];

        result = result + fluidAlgorithm(eDim::y, pars, ypp, yp, cur, yn, ynn);
    }

    return Fraction(result.E, result.R, make_float3(result.Vx, result.Vy, result.Vz));
}
