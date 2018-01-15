#include "octreemanger.h"



OctreeManger::OctreeManger(StartArgs args) : 
    tree(args.X_SIZE, nullptr)
{
}


OctreeManger::~OctreeManger()
{

}

void OctreeManger::insertFract(Fraction * fract, int x, int y, int z)
{
    fractList.push_back(FractionPtr(fract));
    tree.set(x, y, z, fract);
}
