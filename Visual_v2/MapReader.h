#include <string>
#include <fstream>
#include <vector>
#include <windows.h>

class Point
{
public:
	float x, y, z;
	float intensity;
	float flux;
};

class Iteration
{
public:
	float minIntensity, maxIntensity, minFlux, maxFlux;
	unsigned int elementsNum;
	Point *point;
};

class Iterations
{
public:
	float minIntensity, maxIntensity, minFlux, maxFlux;
	unsigned int sizeX, sizeY, sizeZ;
	unsigned int IterationNum;
	Iteration *iteration;
};

class MapReader
{
private:
	Iterations*	_iterations;
	char *pathC;
	bool fluxParam;

	int index = 0, local_x = 0, local_y = 0, currentIter = 0;
	float maxInten = -1.0f, minInten = 10000.0f;

public:
	MapReader(char *path, bool fluxParam);
	~MapReader();
	Iterations* MapReader::GetIterations();

private:
	void MapReader::ReadFile();
	void MapReader::Normalize();
	void MapReader::NormalizeGlobal();
	bool MapReader::GetHeader(std::string line);
	bool MapReader::InterprateLine(std::string line);
	unsigned int MapReader::split(const std::string &txt, std::vector<std::string> &strs, char ch);

};
