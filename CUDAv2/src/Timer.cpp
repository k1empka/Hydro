#include "Timer.h"
#include <iostream>

Timer &Timer::getInstance()
{
    static Timer instance;
    return instance;
}

void Timer::start(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);

    if (it == resultMap.end())
        resultMap[name] = ResultsData(0, timeNow());
    else
    {
        resultMap[name].startPoint = timeNow();
    }
}

long long Timer::stop(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);
    long long dur = 0;
    if (it != resultMap.end())
    {
        dur = duration(timeNow() - resultMap[name].startPoint);
        resultMap[name].resultsSum += dur;
        resultMap[name].count++;
    }
    return dur;
}

long long Timer::stop()
{
    return duration(timeNow() - startTime);
}


void Timer::printResults()
{
    for (ResultMapIt it = resultMap.begin(); it != resultMap.end(); ++it)
    {
        long long avarage = it->second.resultsSum / it->second.count;
        std::cout << it->first.c_str() << ":\n" <<
            "    Avarage execution time: " << avarage / 1000. << "s\n" <<
            "    Total execution time: " << it->second.resultsSum / 1000. << "s\n" <<
            "    Executions count: " << it->second.count << "\n";
    }
}

void Timer::updateMap(std::string const name, long long value)
{
    ResultMapIt it = resultMap.find(name);

    if (it == resultMap.end())
        resultMap[name] = ResultsData(1, timeNow());
    else
    {
        resultMap[name].count++;
        resultMap[name].resultsSum += value;
    }
}

long long Timer::getAvgResult(std::string const& name)
{
    ResultMapIt it = resultMap.find(name);
    if (it == resultMap.end())
           return 0;
    else
    {
        long long avarage = it->second.resultsSum / it->second.count;
        return avarage;
    }
}
