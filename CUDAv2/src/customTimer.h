#pragma once

#include <chrono>
#include <utility>
#include <map>

#define timeNow() std::chrono::high_resolution_clock::now()
#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
//#define MEASURE_MODE 1

struct ResultsData;
typedef std::chrono::high_resolution_clock::time_point TimePoint;
typedef std::map<std::string, ResultsData> ResultMap;
typedef ResultMap::iterator ResultMapIt;

struct ResultsData
{
    long      count;
    long long resultsSum;
    TimePoint startPoint;

    ResultsData() {}
    ResultsData(long count, TimePoint startPoint) : count(count), startPoint(startPoint) { resultsSum = 0;  }
};

class Timer
{

public:
    static Timer&      getInstance();
    void               start(std::string const& title);
    long long               stop(std::string const& title);
    long long          stop();
    void               printResults();
    ResultMap          getResultMap() { return resultMap; }
    long long          getAvgResult(std::string const& name);
    void               clear() { resultMap.clear(); }
    
    template<typename RetType>
    RetType    measure(std::string name, RetType(*method)(void));
    template<typename ObjectType, typename RetType, typename ...Args>
    RetType measure(std::string name, ObjectType& object, RetType (ObjectType::*method)(Args...), Args &&...args);
    template<typename RetType, typename ...Args>
    RetType measure(std::string name, RetType (*method)(Args...), Args &&...args);
    template<typename ...Args>
    void    measure(std::string name, void(*method)(Args...), Args &&...args);
    
private:
    TimePoint startTime;
    ResultMap resultMap;

    void      updateMap(std::string const name, long long value);

              Timer() {};
             ~Timer() {};
};

template<typename ObjectType, typename RetType, typename ...Args>
inline RetType Timer::measure(std::string name, ObjectType& object, RetType(ObjectType::*method)(Args...), Args &&...args)
{
    TimePoint start = timeNow();

    auto ret = (object.*method)(std::forward<Args>(args)...);

    updateMap(name, duration(timeNow() - start));
    return ret;
}

template<typename RetType, typename ...Args>
inline RetType Timer::measure(std::string name, RetType (*method)(Args...), Args && ...args)
{
    TimePoint start = timeNow();

    auto ret = *method(std::forward<Args>(args)...);

    updateMap(name, duration(timeNow() - start));
    return ret;
}

template<typename ...Args>
inline void Timer::measure(std::string name, void(*method)(Args...), Args && ...args)
{
    TimePoint start = timeNow();

    method(std::forward<Args>(args)...);

    updateMap(name, duration(timeNow() - start));
}

template<typename RetType>
inline RetType Timer::measure(std::string name, RetType(*method)(void))
{
    TimePoint start = timeNow();

    auto ret = method();
    updateMap(name, duration(timeNow() - start));
    return ret;
}
