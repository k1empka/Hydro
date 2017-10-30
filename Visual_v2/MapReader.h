#include <string>
#include <fstream>
#include <vector>
#include <windows.h>

class Point
{
public:
	float x, y, z;
	float intensity;
};

class Iteration
{
public:
	float minIntensity, maxIntensity;
	unsigned int elementsNum;
	Point *point;
};

class Iterations
{
public:
	unsigned int sizeX, sizeY, sizeZ;
	unsigned int IterationNum;
	Iteration *iteration;
};

class MapReader
{
private:
	Iterations*	_iterations;
	char *pathC;

	int index = 0, local_x = 0, local_y = 0, currentIter = 0;
	float maxInten = -1.0f, minInten = 10000.0f;

public:
	MapReader(char *path);
	~MapReader();
	Iterations* MapReader::GetIterations();

private:
	void MapReader::ReadFile();
	bool MapReader::GetHeader(std::string line);
	bool MapReader::InterprateLine(std::string line);
	unsigned int MapReader::split(const std::string &txt, std::vector<std::string> &strs, char ch);

};
