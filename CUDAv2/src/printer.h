#pragma once
#include <stdlib.h>
#include <stdio.h>

#include "computation.h"

class Printer
{
public:
    Printer(const char* title, const StartArgs args);
    ~Printer();
    void printIteration(Fraction* space, int iter);

private:
    FILE* f = nullptr;
    int X_SIZE;
    int Y_SIZE;
    int Z_SIZE;
    int NUM_OF_ITERATIONS;

    void initOutputFile(const char* title);
    void printHeader();
};

