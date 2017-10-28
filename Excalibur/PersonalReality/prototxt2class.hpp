#pragma once
#include "readcaffemodel.hpp"
#include <iostream>

class prototxt2class
{
	std::ofstream cpp;
	std::ofstream hpp; 
	std::map<int, std::string> net_dag;
public:
	prototxt2class(std::string outpath, std::string netname);
	~prototxt2class();

	void declear_params(std::string caffemodel, std::string netname);

	void declear_operation_neuron(std::string prorotxt);

	void delete_operation(std::string prorotxt);

	void build_net_dag(std::string prorotxt);

	void init_operation(std::string prorotxt);

	void build_forward(std::string prorotxt);
};

