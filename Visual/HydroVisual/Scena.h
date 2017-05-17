#pragma once
#include <windows.h>
#include <fstream>
#include <string>
#include <vector>

#include "Cube.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

class Scena
{
	LPDIRECT3D9 d3d;    // Direct3D interface
	LPDIRECT3DDEVICE9 d3ddev;    // device
	LPDIRECT3DVERTEXBUFFER9 v_buffer; //buffer
	LPDIRECT3DINDEXBUFFER9 i_buffer; //buffer
	 
	Cube *cubes;
	IterationStruct *iterations;

	ID3DXFont *font;
	RECT fRectangle;
	std::string message = "0";

public:
	std::ifstream myfile;
	int SIZE_X = 100;
	int SIZE_Y = 100;
	int currentIter = 0;
	unsigned int iterationNum = 0;

	void init3D(HWND hWnd);
	void initIterations(char *path);
	void renderFrame();
	void cleanD3D();
	void cleanBuff();

private:
	void setCamera();
	unsigned int split(const std::string &txt, std::vector<std::string> &strs, char ch);
};