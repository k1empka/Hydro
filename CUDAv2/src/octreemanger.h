#pragma once
#include <list>

#include "Fraction.h"
#include "octree\octree.h"
#include "computation.h"

class OctreeManger
{
public:
    OctreeManger(StartArgs args);
    ~OctreeManger();

    void insertFract(Fraction* fract, int x, int y, int z);

private:
    std::list<FractionPtr> fractList;
    Octree<Fraction*> tree;

};

