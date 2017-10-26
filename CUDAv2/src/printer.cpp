#include <stdint.h>

#include "printer.h"

Printer::Printer(const char* title)
{
    initOutputFile(title);
    printHeader();
    printf("Data written to: %s\n", title);
}

Printer::~Printer()
{
    fclose(f);
}

void Printer::initOutputFile(const char* title)
{
    f = fopen(title, "wb");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
}

void Printer::printHeader()
{
    int16_t x = (int16_t)X_SIZE;
    int16_t y = (int16_t)Y_SIZE;
    int16_t z = (int16_t)Z_SIZE;
    int16_t i = (int16_t)NUM_OF_ITERATIONS;
    int16_t floatSize = (int16_t)sizeof(float);
    int size = sizeof(int16_t);
    fwrite(&x, size, 1, f);
    fwrite(&y, size, 1, f);
    fwrite(&z, size, 1, f);
    fwrite(&i, size, 1, f);
    fwrite(&floatSize, size, 1, f);
}

void Printer::printIteration(fraction* space, int iter)
{
	float v;
	int size = sizeof(float);

	for(int z=0; z<Z_SIZE;++z)
	{
		for(int y=0; y<Y_SIZE;++y)
		{
			for(int x=0; x<X_SIZE;++x)
			{
				v =space->U[IDX_3D(x,y,z)];
				fwrite(&v,size,1,f);
			}
		}
	}
}