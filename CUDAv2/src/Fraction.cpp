#include "Fraction.h"

#include <cmath>

Fraction::Fraction() : E(0), R(0), Vx(0), Vy(0), Vz(0)
{
}

Fraction::Fraction(float _E, float _R, float3 _v) : E(_E), R(_R), Vx(_v.x), Vy(_v.y), Vz(_v.z)
{
}

Fraction::Fraction(float _E, float _R, float vx, float vy, float vz)
 : E(_E), R(_R), Vx(vx), Vy(vy), Vz(vz)
{
}

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
        FM =  make_float3(M.x* v.x, M.y* v.x, M.z* v.x);;
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
