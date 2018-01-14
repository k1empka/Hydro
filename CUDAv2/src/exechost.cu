#include "exec.h"

Fraction* execHost(StartArgs args)
{
    Timer::getInstance().clear();
    int totalSize = sizeof(Fraction) * args.SIZE(), i;
    Printer* bytePrinter = NULL;
    void* space, *result = new Fraction[totalSize];
    auto params = initParams();
    if (NULL == result)
    {
        printf("Malloc problem!\n");
        exit(-1);
    }

    space = initSpace(args);
    if (args.print)
        bytePrinter = new Printer("host.data", args);

    printf("Host simulation started\n");
    Timer::getInstance().start("Host simulation time");
    for (i = 0; i<args.NUM_OF_ITERATIONS; ++i)
    {
        hostSimulation(args, &params, space, result);
        swapPointers(space, result);
        if (args.print)
            if (i % 2 == 0)
                bytePrinter->printIteration((Fraction*)space, i);
            else
                bytePrinter->printIteration((Fraction*)result, i);
    }
    Timer::getInstance().stop("Host simulation time");
    printf("Host simulation completed\n");
    Timer::getInstance().printResults();

    if (bytePrinter) delete bytePrinter;

    if (i % 2 == 0)
    {
        free(result);
        return (Fraction*)space;
    }
    else
    {
        free(space);
        return (Fraction*)result;
    }
}

Fraction* execHostOctree(StartArgs args)
{
    return nullptr;
}
