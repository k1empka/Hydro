#pragma once

#include <cuda_runtime.h>
#include <memory>

class Fraction;
typedef std::shared_ptr<Fraction> FractionPtr;

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
    __host__ __device__ Fraction() 
        : E(0), R(0), Vx(0), Vy(0), Vz(0) {};
    __host__ __device__ Fraction(float _E, float _R, float3 v)
        : E(_E), R(_R), Vx(v.x), Vy(v.y), Vz(v.z) {}
    __host__ __device__ Fraction(float _E, float _R, float vx, float vy, float vz)
        : E(_E), R(_R), Vx(vx), Vy(vy), Vz(vz) {}
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