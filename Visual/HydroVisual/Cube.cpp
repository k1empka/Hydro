#include "Cube.h"

Cube::Cube()
{ }

void Cube::InitObject(LPDIRECT3DDEVICE9 dev, float x, float y, int rc, int bc, int gc)
{
	x *= 2.0f;
	y *= 2.0f;

	CustomVertex vertices[] =
	{ 
		{ -1.0f-x, 1.0f-y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },    // vertex 0
		{ 1.0f-x, 1.0f-y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },     // vertex 1
		{ -1.0f-x, -1.0f-y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },   // 2
		{ 1.0f-x, -1.0f-y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },  // 3
		{ -1.0f-x, 1.0f-y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },     // ...
		{ 1.0f-x, 1.0f-y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ -1.0f-x, -1.0f-y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ 1.0f-x, -1.0f-y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
	};

	dev->CreateVertexBuffer((sizeof(vertices) / sizeof(*vertices)) * sizeof(CustomVertex),
		0,
		CUSTOMFVF,
		D3DPOOL_MANAGED,
		&v_buffer,
		NULL);
	
	VOID* pVoid;
	v_buffer->Lock(0, 0, (void**)&pVoid, 0);
	memcpy(pVoid, vertices, sizeof(vertices));
	v_buffer->Unlock();

	short indices[] =
	{
		0, 1, 2,    // side 1
		2, 1, 3,
		4, 0, 6,    // side 2
		6, 0, 2,
		7, 5, 6,    // side 3
		6, 5, 4,
		3, 1, 7,    // side 4
		7, 1, 5,
		4, 5, 0,    // side 5
		0, 5, 1,
		3, 7, 2,    // side 6
		2, 7, 6,
	};

	dev->CreateIndexBuffer(36 * sizeof(short),
		0,
		D3DFMT_INDEX16,
		D3DPOOL_MANAGED,
		&i_buffer,
		NULL);
	
	i_buffer->Lock(0, 0, (void**)&pVoid, 0);
	memcpy(pVoid, indices, sizeof(indices));
	i_buffer->Unlock();
}

void Cube::UpdateColor(float x, float y, int rc, int bc, int gc)
{
	x *= 2.0f;
	y *= 2.0f;

	CustomVertex vertices[] =
	{
		{ -1.0f - x, 1.0f - y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ 1.0f - x, 1.0f - y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ -1.0f - x, -1.0f - y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ 1.0f - x, -1.0f - y, -1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ -1.0f - x, 1.0f - y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ 1.0f - x, 1.0f - y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ -1.0f - x, -1.0f - y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
		{ 1.0f - x, -1.0f - y, 1.0f, D3DCOLOR_XRGB(rc, gc, bc), },
	};

	VOID* pVoid;
	v_buffer->Lock(0, 0, (void**)&pVoid, 0);
	memcpy(pVoid, vertices, sizeof(vertices));
	v_buffer->Unlock();
}