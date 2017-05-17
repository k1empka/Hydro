#include "CustomResources.h"

#define INSTANCE_SIZE 5

class Cube
{
public:
	LPDIRECT3DVERTEXBUFFER9 v_buffer = NULL;
	LPDIRECT3DINDEXBUFFER9 i_buffer = NULL;

	Cube();
	void InitObject(LPDIRECT3DDEVICE9 dev, float x, float y, int rc, int bc, int gc);
	void UpdateColor(float x, float y, int rc, int bc, int gc);
};