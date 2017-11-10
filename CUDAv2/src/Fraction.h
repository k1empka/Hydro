#pragma once

#include <cuda_runtime.h>

enum class eDim
{
    x,
    y,
    z
};

struct vp
{
    float p;
    float v;
};

class Fraction
{
public:
    __host__ __device__ Fraction();
    __host__ __device__ Fraction(float E,float R,float3 v);
    __host__ __device__ Fraction(float E, float R, float vx,float vy,float vz);

    __host__ __device__ Fraction  flux(eDim dim);
    __host__ __device__ vp        vnep_calc();
    __host__ __device__ Fraction  operator+(Fraction const& f)
    {
        return Fraction(E + f.E,R + f.R,f.Vx + Vx, f.Vy + Vy, f.Vz + Vz);
    }
    __host__ __device__ Fraction  operator-(Fraction const& f)
    {
        return Fraction(E - f.E, R - f.R, Vx - f.Vx, Vy - f.Vy,Vz - f.Vz);
    }
    __host__ __device__ Fraction  operator*(Fraction const& f)
    {
        return Fraction(E * f.E, R * f.R, Vx * f.Vx, Vy * f.Vy, Vz * f.Vz);
    }
    __host__ __device__ Fraction  operator/(Fraction const& f)
    {
        return Fraction(E / f.E, R / f.R, Vx / f.Vx, Vy / f.Vy, Vz / f.Vz);
    }
    __host__ __device__ Fraction  operator*(float num)
    {
        return Fraction(E * num, R * num, Vx * num, Vy * num, Vz * num);
    }
    __host__ __device__ Fraction  operator-(float num)
    {
        return Fraction(E - num, R - num, Vx - num, Vy - num, Vz - num);
    }
    __host__ __device__ Fraction  operator+(float num)
    {
        return Fraction(E + num, R + num, Vx + num, Vy + num, Vz + num);
    }
    __host__ __device__ Fraction  operator/(float num)
    {
        return Fraction(E / num, R / num, Vx / num, Vy / num, Vz / num);
    }

    float  E;
    float  R;
    float  Vx;
    float  Vy;
    float  Vz;
};

struct Fraction2
{
    Fraction L;
    Fraction R;
};