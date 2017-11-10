#include <stdint.h>

#include "printer.h"
#include "computation.h"

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

void Printer::printIteration(Fraction* space, int iter)
{
	float v;
	int size = sizeof(float);

	for(int i=0; i<SIZE;++i)
	{
		v = space[i].E;
		fwrite(&v,size,1,f);					
	}
}