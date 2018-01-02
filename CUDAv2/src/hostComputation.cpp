#include "computation.h"
#include "Fraction.h"

#include <stdio.h>

void hostSimulation(StartArgs args, FluidParams* pars,void* spaceData, void* resultData)
{
	Fraction* spaceFraction = (Fraction*)spaceData;
	Fraction* resultFraction = (Fraction*)resultData;
	for(int z=2; z<args.Z_SIZE-2;++z)
	{
		for(int y=2; y<args.Y_SIZE-2;++y)
		{
			for(int x=2; x<args.X_SIZE-2;++x)
			{
                auto const& f = result3D(args,pars, spaceFraction, make_int3(x, y, z));
                resultFraction[args.IDX_3D(x, y, z)] = f;
			}
		}
	}
}
