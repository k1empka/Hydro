#pragma once
#include <d3d9.h>
#include <d3dx9.h>

#define CUSTOMFVF (D3DFVF_XYZ | D3DFVF_DIFFUSE)

struct CustomVertex
{
	FLOAT x, y, z;    // from the D3DFVF_XYZRHW flag
	DWORD color;    // from the D3DFVF_DIFFUSE flag
};

struct Point
{
	FLOAT x, y;
};

struct IterationStruct
{
	unsigned int elementsNum;
	Point *point;
};