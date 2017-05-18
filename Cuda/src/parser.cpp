/*
 * Parser.cpp
 *
 *  Created on: 14 maj 2017
 *      Author: mknap
 */

#include "parser.h"
#include "particle.h"

#include <iostream>

Parser::Parser(char* inPath,char* outPath)
{
	in.open(inPath);
	out.open(outPath);
	writeHeader();
}
Parser::~Parser()
{
	in.close();
	out.close();
}

bool Parser::readEntryData(char* path,Particle* data)
{
	/*std::ifstream inFile;
	inFile.open(path);

	if(inFile.is_open())
	{
		int sX, sY, sZ;
		std::cin >> sX >> sY >> sZ;
	}
	*/

	//TO DO
	return true;
}

void Parser::writeHeader()
{
	if(out.is_open())
	{
		out << Factors::X_SIZE << " " <<
			   Factors::Y_SIZE << " " <<
			   //Factors::Z_SIZE << " " <<
			   Factors::ITERATION_NUM+1 << std::endl;
	}
	else
	{
		std::cout<< "Can't open output file\n";
		exit(1);
	}
}

void Parser::writeIterToFile3D(Particle* data,int iter)
{
	out << "ITER_" << iter << std::endl;
	for(int i = 0; i < Factors::TOTAL_SIZE; ++i)
	{
		float4 pos = data->position[i];
		float4 v   = data->velocity[i];

		if(pos.x > 0 || pos.y > 0 || pos.z > 0)
		{
			out << pos.x << " " << pos.y << " " << pos.z << " "
				<< v.x   << " " << v.y   << " " << v.z   << std::endl;
		}
	}
}
void Parser::writeIterToFile2D(Particle* data,int iter)
{
	out << "ITER_" << iter << std::endl;
	for(int i = 0; i < Factors::TOTAL_SIZE; ++i)
	{
		float4 pos = data->position[i];
		float4 v   = data->velocity[i];

		if(pos.x > 0 || pos.y > 0)
		{
			out << pos.x << " " << pos.y << " "
				<< v.x   << " " << v.y   << " " << std::endl;
		}
	}
}
