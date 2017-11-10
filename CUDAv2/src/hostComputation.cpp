#include "computation.h"
#include "Fraction.h"

void hostSimulation(FluidParams* pars,void* spaceData, void* resultData)
{
	Fraction* spaceFraction = (Fraction*)spaceData;
	Fraction* resultFraction = (Fraction*)resultData;

	for(int z=2; z<Z_SIZE-2;++z)
	{
		for(int y=2; y<Y_SIZE-2;++y)
		{
			for(int x=2; x<X_SIZE-2;++x)
			{
              //  resultFraction[IDX_3D(x,y,z)] = result3D(pars,spaceFraction,make_int3(x,y,z));
			}
		}
	}
}
