/*
 * Parser.h
 *
 *  Created on: 14 maj 2017
 *      Author: mknap
 */

#ifndef PARSER_H_
#define PARSER_H_

#include <fstream>

#include "particle.h"

class Parser
{
public:
	Parser(char* inPath, char* outPath);
	~Parser();
	bool readEntryData(char* path,Particle* data);
	void writeIterToFile3D(Particle* data,int iter);
	void writeIterToFile2D(Particle* data,int iter);

private:
	std::ifstream in;
	std::ofstream out;

	void writeHeader();
};

#endif /* PARSER_H_ */
