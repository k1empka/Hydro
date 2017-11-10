#pragma once
#include <stdlib.h>
#include <stdio.h>

#include "Fraction.h"

class Printer
{
public:
    Printer(const char* title);
    ~Printer();
    void printIteration(Fraction* space, int iter);

private:
    FILE* f = nullptr;

    void initOutputFile(const char* title);
    void printHeader();
};

