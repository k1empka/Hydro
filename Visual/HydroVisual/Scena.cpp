#include "Scena.h"

void Scena::init3D(HWND hWnd)
{
	d3d = Direct3DCreate9(D3D_SDK_VERSION);

	D3DPRESENT_PARAMETERS d3dpp;

	ZeroMemory(&d3dpp, sizeof(d3dpp));
	d3dpp.Windowed = TRUE;
	d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
	d3dpp.hDeviceWindow = hWnd;
	d3dpp.BackBufferFormat = D3DFMT_X8R8G8B8;
	d3dpp.BackBufferWidth = SCREEN_WIDTH;
	d3dpp.BackBufferHeight = SCREEN_HEIGHT;
	d3dpp.EnableAutoDepthStencil = TRUE;
	d3dpp.AutoDepthStencilFormat = D3DFMT_D16;

	d3d->CreateDevice(
		D3DADAPTER_DEFAULT,
		D3DDEVTYPE_HAL,
		hWnd,
		D3DCREATE_SOFTWARE_VERTEXPROCESSING,
		&d3dpp,
		&d3ddev
	);

	D3DXCreateFont(d3ddev, 25, 0, FW_NORMAL, 1, false, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, ANTIALIASED_QUALITY, FF_DONTCARE, "Arial", &font);
	SetRect(&fRectangle, 0, 0, 100, 30);

	d3ddev->SetRenderState(D3DRS_LIGHTING, FALSE);
	d3ddev->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE); 
	d3ddev->SetRenderState(D3DRS_ZENABLE, TRUE); // 3D - Z axis

	return;
}

void Scena::renderFrame()
{
	DWORD start = GetTickCount();	
	for (int i = 0; i < iterations[currentIter].elementsNum; i++)
	{
		if(currentIter==0)
			cubes[i].InitObject(d3ddev,iterations[currentIter].point[i].x, iterations[currentIter].point[i].y, 0, 255, 0);
		else
			cubes[i].UpdateColor(iterations[currentIter].point[i].x, iterations[currentIter].point[i].y, 0, 255, 0);
	}

	d3ddev->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 0, 0), 1.0f, 0);
	d3ddev->Clear(0, NULL, D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(0, 0, 0), 1.0f, 0);

	d3ddev->BeginScene();

	d3ddev->SetFVF(CUSTOMFVF);
	setCamera();

	for (int i = 0; i < iterations[currentIter].elementsNum;i++)
	{
		d3ddev->SetStreamSource(0, cubes[i].v_buffer, 0, sizeof(CustomVertex));
		d3ddev->SetIndices(cubes[i].i_buffer);
		d3ddev->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, 8, 0, 12);
	} 
	d3ddev->EndScene();
	d3ddev->Present(NULL, NULL, NULL, NULL);
	
	DWORD stop = GetTickCount();
	DWORD delta;
	if (stop - start > 0)
		delta = 1000 / (stop - start);
	else
		delta = 1;
	
	// WARNING, DANGER!! -> SLEEP! 
	// *
	if (delta < 66) // 66 -> 15 fps
	{
		Sleep(66 - delta);
		delta = 66;
	}
	// *

	//DWORD fps = 1000 / delta;
	message = std::to_string(1000 / delta) + " FPS";
	
	// when we have last iteration, just draw the same previous frame
	if(currentIter<iterationNum-2) 
		currentIter++;
}

void Scena::cleanD3D()
{
	d3ddev->Release();
	d3d->Release();
	font->Release();
}

void Scena::initIterations(char *path)
{

	bool sizeReaded = false;

	std::string line;
	std::ifstream myfile(path);

	int index = 0;
	int currentIter = 0;
	while (getline(myfile, line))
	{
		std::vector<std::string> vec;
		UINT size = split(line, vec, ' ');
		if (!sizeReaded)
		{
			SIZE_X = std::atoi(vec.at(0).c_str());
			SIZE_Y = std::atoi(vec.at(1).c_str());
			iterationNum = std::stoul(vec.at(2), nullptr, 0);
			unsigned int sizeMap = SIZE_X*SIZE_Y;
			cubes = new Cube[sizeMap];
			iterations = new IterationStruct[iterationNum];
			sizeReaded = true;
			for (int i = 0; i < iterationNum; i++)
			{
				iterations[i].point = new Point[sizeMap];
			}
			continue;
		}
		if (vec.at(0).find("ITER_") != std::string::npos)
		{
			iterations[currentIter].elementsNum = index;
			index = 0;
			currentIter = std::atoi(vec.at(0).substr(5, 6).c_str());
			continue;
		}
		if (vec.size() == 5)
		{
			float x = (float)std::atof(vec.at(0).c_str());
			float y = (float)std::atof(vec.at(1).c_str());
			iterations[currentIter].point[index].x = x;
			iterations[currentIter].point[index].y = y;
			index++;
		}
	}
	myfile.close();

}

void Scena::cleanBuff()
{
	for (int i = 0; i < SIZE_X*SIZE_Y; i++)
	{
		cubes[i].v_buffer->Release();
		cubes[i].i_buffer->Release();
	}
	
	delete[] cubes;
}

void Scena::setCamera()
{
	DWORD distance = SIZE_X + 20.0f;

	D3DXMATRIX matView;    
	D3DXMatrixLookAtLH(&matView,
		&D3DXVECTOR3(-((float)SIZE_X), -((float)SIZE_Y), distance),
		&D3DXVECTOR3(-((float)SIZE_X), -((float)SIZE_Y), 0.0f),
		&D3DXVECTOR3(0.0f, 1.0f, 0.0f)); 
	d3ddev->SetTransform(D3DTS_VIEW, &matView);  
	
	D3DXMATRIX matProjection;  
	D3DXMatrixPerspectiveFovLH(&matProjection,
		D3DXToRadian(90),
		(FLOAT)SCREEN_WIDTH / (FLOAT)SCREEN_HEIGHT,
		distance-5,
		distance+5);
	d3ddev->SetTransform(D3DTS_PROJECTION, &matProjection);
	
	font->DrawTextA(NULL, message.c_str(), -1, &fRectangle,DT_LEFT,D3DCOLOR_XRGB(255,255,0));

	/*
	//D3DXMATRIX matTranslateA;
	//D3DXMATRIX matTranslateB;
	D3DXMATRIX matRotateY;
	D3DXMATRIX matRotateX;
	static float index = 0.0f; index += 0.01f;

											   

	D3DXMatrixRotationY(&matRotateY, index);  
	D3DXMatrixRotationX(&matRotateX, index);

	d3ddev->SetTransform(D3DTS_WORLD, &(matRotateY * matRotateX));
	d3ddev->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
	*/
}

unsigned int Scena::split(const std::string &txt, std::vector<std::string> &strs, char ch)
{
	unsigned int pos = txt.find(ch);
	unsigned int initialPos = 0;
	strs.clear();

	// the piece of code 'pos != 4294967295' must be change 
	while (pos != std::string::npos && pos != 4294967295) {
		strs.push_back(txt.substr(initialPos, pos - initialPos + 1));
		initialPos = pos + 1;

		pos = txt.find(ch, initialPos);
	}

	if(pos < txt.size())
		strs.push_back(txt.substr(initialPos, pos - initialPos + 1));
	else
		strs.push_back(txt.substr(initialPos, txt.size() - initialPos + 1));

	return strs.size();
}