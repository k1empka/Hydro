#include "MapReader.h"

MapReader::MapReader(char *path, bool fluxParam = false)
{
	this->_iterations = new Iterations();
	this->_iterations->maxFlux = -1000000.0f;
	this->_iterations->maxIntensity = -1000000.0f;
	this->_iterations->minFlux = 1000000.0f;
	this->_iterations->minIntensity = 1000000.0f;
	this->pathC = path;
	this->fluxParam = fluxParam;

	ReadFile();
}

MapReader::~MapReader()
{
	delete this->_iterations;
}

void MapReader::ReadFile()
{
	std::string line;
	std::ifstream myfile(this->pathC, std::ios::binary);
	
	INT16 storage[5];
	float cell[2];

	if (myfile.is_open())
	{
		bool headerReaded = false;
		myfile.read((char*)&storage, 5 * sizeof(INT16));

		this->_iterations->sizeX = storage[0];
		this->_iterations->sizeY = storage[1];
		this->_iterations->sizeZ = storage[2];
		this->_iterations->IterationNum = storage[3] - 1;
		int size_cell = storage[4];
		this->_iterations->iteration = new Iteration[this->_iterations->IterationNum];
		for (int i = 0; i < this->_iterations->IterationNum; i++)
			this->_iterations->iteration[i].point = new Point[storage[0] * storage[1] * storage[2]];

		for (int iter = 0; iter < this->_iterations->IterationNum; iter++)
		{
			float maxInt = -100000.0f;
			float minInt = 100000.0f;
			float maxFlu = -1000000.0f;
			float minFlu = 1000000.0f;
			for (int z = 0; z < this->_iterations->sizeZ; z++)
			{
				for (int y = 0; y < this->_iterations->sizeY; y++)
				{
					for (int x = 0; x < this->_iterations->sizeX; x++)
					{
						if (this->fluxParam)
						{
							myfile.read((char*)(&cell), sizeof(size_cell) * 2);
							this->_iterations->iteration[iter].point[z*(this->_iterations->sizeY*this->_iterations->sizeX) + y*this->_iterations->sizeY + x].flux = cell[1];
						}
						else
							myfile.read((char*)(&cell), sizeof(size_cell));
						if (z >= 95)
							int i = 0;
						this->_iterations->iteration[iter].point[z*(this->_iterations->sizeY*this->_iterations->sizeX)+y*this->_iterations->sizeY + x].x = x;
						this->_iterations->iteration[iter].point[z*(this->_iterations->sizeY*this->_iterations->sizeX) + y*this->_iterations->sizeY + x].y = y;
						this->_iterations->iteration[iter].point[z*(this->_iterations->sizeY*this->_iterations->sizeX) + y*this->_iterations->sizeY + x].z = z;
						this->_iterations->iteration[iter].point[z*(this->_iterations->sizeY*this->_iterations->sizeX) + y*this->_iterations->sizeY + x].intensity = cell[0];

						if (cell[0] > maxInt)
							maxInt = cell[0];
						if (cell[0] <= minInt)
							minInt = cell[0];

						if (cell[1] > maxFlu)
							maxFlu = cell[1];
						if (cell[1] <= minFlu)
							minFlu = cell[1];
					}
				}
			}
			this->_iterations->iteration[iter].elementsNum = this->_iterations->sizeX*this->_iterations->sizeY;
			this->_iterations->iteration[iter].maxIntensity = maxInt;
			this->_iterations->iteration[iter].minIntensity = minInt;
			this->_iterations->iteration[iter].maxFlux = maxFlu;
			this->_iterations->iteration[iter].minFlux = minFlu;

			if (maxInt > this->_iterations->maxIntensity)
				this->_iterations->maxIntensity = maxInt;
			if (minInt <= this->_iterations->minIntensity)
				this->_iterations->minIntensity = minInt;

			if (maxFlu > this->_iterations->maxFlux)
				this->_iterations->maxFlux = maxFlu;
			if (minFlu <= this->_iterations->minFlux)
				this->_iterations->minFlux = minFlu;
		}

		Normalize();
		//NormalizeGlobal();
	}
	else
	{
		throw std::invalid_argument("File is not exists!");
	}
}

void MapReader::Normalize()
{
	for (int iter = 0; iter < this->_iterations->IterationNum; iter++)
	{
		float minFluxAbs = 0.0f;
		float minIntensityAbs = 0.0f;

		if (this->fluxParam)
		{
			minFluxAbs = (this->_iterations->iteration[iter].minFlux < 0 ? abs(this->_iterations->iteration[iter].minFlux) : this->_iterations->iteration[iter].minFlux*(-1));
			this->_iterations->iteration[iter].minFlux = 0.0f;
			this->_iterations->iteration[iter].maxFlux += minFluxAbs;
		}

		minIntensityAbs = (this->_iterations->iteration[iter].minIntensity < 0 ? abs(this->_iterations->iteration[iter].minIntensity) : this->_iterations->iteration[iter].minIntensity*(-1));
		this->_iterations->iteration[iter].minIntensity = 0.0f;
		this->_iterations->iteration[iter].maxIntensity += minFluxAbs;

		for (int index = 0; index < this->_iterations->sizeZ*this->_iterations->sizeY*this->_iterations->sizeX; index++)
		{
			this->_iterations->iteration[iter].point[index].flux += minFluxAbs;
			this->_iterations->iteration[iter].point[index].intensity += minIntensityAbs;
		}
	}
}

// Normalize negative values for intensity and flux
void MapReader::NormalizeGlobal()
{
	float minFluxAbs = (this->_iterations->minFlux < 0 ? abs(this->_iterations->minFlux) : this->_iterations->minFlux*(-1));
	float minIntensityAbs = (this->_iterations->minIntensity < 0 ? abs(this->_iterations->minIntensity) : this->_iterations->minIntensity*(-1));
	for (int iter = 0; iter < this->_iterations->IterationNum; iter++)
	{
		float minFluxAbsIter = 0.0f;
		if (this->fluxParam)
		{
			minFluxAbsIter = (this->_iterations->iteration[iter].minFlux < 0 ? abs(this->_iterations->iteration[iter].minFlux) : this->_iterations->iteration[iter].minFlux*(-1));
			this->_iterations->iteration[iter].minFlux = 0.0f;
			this->_iterations->iteration[iter].maxFlux += minFluxAbsIter;
		}

		float minIntensityAbsIter = (this->_iterations->iteration[iter].minIntensity < 0 ? abs(this->_iterations->iteration[iter].minIntensity) : this->_iterations->iteration[iter].minIntensity*(-1));
		this->_iterations->iteration[iter].minIntensity = 0.0f;
		this->_iterations->iteration[iter].maxIntensity += minIntensityAbsIter;

		for (int index = 0; index < this->_iterations->sizeZ*this->_iterations->sizeY*this->_iterations->sizeX; index++)
		{
			this->_iterations->iteration[iter].point[index].flux += minFluxAbs;
			this->_iterations->iteration[iter].point[index].intensity += minIntensityAbs;
		}
	}
	this->_iterations->maxFlux += minFluxAbs;
	this->_iterations->minFlux += minFluxAbs;
	this->_iterations->maxIntensity += minIntensityAbs;
	this->_iterations->minIntensity += minIntensityAbs;
}

Iterations* MapReader::GetIterations()
{
	return this->_iterations;
}

bool MapReader::GetHeader(std::string line)
{
	std::vector<std::string> vec;
	unsigned int count = split(line, vec, ' ');
	if (count != 2)
		throw std::invalid_argument("Header is not supported");
	unsigned int sizeX = std::atoi(vec.at(0).c_str());
	unsigned int sizeY = std::atoi(vec.at(1).c_str());
	unsigned int sizeZ = 0;

	this->_iterations->IterationNum = std::stoul(vec.at(2), nullptr, 0);
	this->_iterations->sizeX = sizeX;
	this->_iterations->sizeY = sizeY;
	this->_iterations->sizeZ = sizeZ;

	this->_iterations->iteration = new Iteration[this->_iterations->IterationNum];
	for (int i = 0; i < this->_iterations->IterationNum; i++)
		this->_iterations->iteration[i].point = new Point[sizeX*sizeY];

	return true;
}

bool MapReader::InterprateLine(std::string line)
{
	std::vector<std::string> vec;
	unsigned int count = split(line, vec, ' ');
	if (vec.at(0).find("ITER_") != std::string::npos)
	{
		this->_iterations->iteration[currentIter].elementsNum = index;
		this->_iterations->iteration[currentIter].maxIntensity = maxInten;
		this->_iterations->iteration[currentIter].minIntensity = minInten;

		index = 0;
		local_x = 0;
		local_y = 0;

		currentIter++;
		maxInten = -1.0f;
		minInten = 10000.0f;

		return true;
	}
	if (count == 4)
	{
		int x = (int)std::atof(vec.at(0).c_str());
		int y = (int)std::atof(vec.at(1).c_str());
		float intensity = (float)std::atof(vec.at(2).c_str());
		if (intensity > maxInten)
			maxInten = intensity;
		if (intensity < minInten)
			minInten = intensity;

		while (local_x != x && local_y != y)
		{
			this->_iterations->iteration[currentIter].point[index].x = local_x;
			this->_iterations->iteration[currentIter].point[index].y = local_y;
			this->_iterations->iteration[currentIter].point[index].intensity = 0.0;

			index++;
			local_x++;
			if ((local_x / this->_iterations->sizeX) == 1)
			{
				local_x = 0;
				if (local_y < this->_iterations->sizeY - 1) local_y++;
			}
		}

		this->_iterations->iteration[currentIter].point[index].x = x;
		this->_iterations->iteration[currentIter].point[index].y = y;
		this->_iterations->iteration[currentIter].point[index].intensity = intensity;

		index++;
		local_x++;
		if ((local_x / this->_iterations->sizeX) == 1)
		{
			local_x = 0;
			if (local_y < this->_iterations->sizeY - 1) local_y++;
		}
		return true;
	}

	throw std::invalid_argument("Header is not supported");
	return false;
}

unsigned int MapReader::split(const std::string &txt, std::vector<std::string> &strs, char ch)
{
	unsigned int pos = txt.find(ch);
	unsigned int initialPos = 0;
	strs.clear();

	// the piece of code 'pos != 4294967295' must be change 
	while (pos != std::string::npos && pos != 4294967295) {
		strs.push_back(txt.substr(initialPos, pos - initialPos + 1));
		initialPos = pos + 1;

		pos = txt.find(ch, initialPos);
	}

	if (pos < txt.size())
		strs.push_back(txt.substr(initialPos, pos - initialPos + 1));
	else
		strs.push_back(txt.substr(initialPos, txt.size() - initialPos + 1));

	return strs.size();
}