#include "datafile.hpp"
#include "prototxt2class.hpp"

#pragma   warning   (disable:4146)

int main()
{
	datafile* df = new datafile("D:/projects/MCL_Forward/Excalibur/Cassius/unicornNet_halfdata.hpp");
	df->writedatahpp("D:/projects/data/unicornModel/unicorn9734_usefulpart.caffemodel", "unicorn_net", "cassius");
	
	//prototxt2class* pc = new prototxt2class("D:\\Research\\MCL_Forward\\Excalibur\\model", "unicorn_net");
	//std::string prototxt = "D:/projects/data/unicornModel/unicorn_deploy.prototxt";
	//pc->declear_params("D:/projects/data/unicornModel/unicorn9734_usefulpart.caffemodel", "unicorn_net");
	//pc->declear_operation_neuron(prototxt);
	//pc->delete_operation(prototxt);
	//pc->build_net_dag(prototxt);
	//pc->init_operation(prototxt);
	//pc->build_forward(prototxt);
	return 0;
}