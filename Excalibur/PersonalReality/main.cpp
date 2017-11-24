#include "datafile.hpp"
#include "prototxt2class.hpp"
int main()
{
	datafile* df = new datafile("D:\\Research\\MCL_Forward\\Excalibur\\model\\unicorn_data.hpp");
	df->writedatahpp("E:\\rec-bench\\model\\centerloss34_usefulpart.caffemodel", "Unicorn", "glasssix");
	
	/*prototxt2class* pc = new prototxt2class("D:\\Research\\MCL_Forward\\Excalibur\\model", "IPTs_v2");
	std::string prototxt = "D:\\Research\\MCL_Forward\\Excalibur\\model\\5IPTs_v2_deploy.prototxt";
	pc->declear_params("D:\\Research\\MCL_Forward\\Excalibur\\model\\5IPTs_v2.caffemodel", "IPTs_v2");
	pc->declear_operation_neuron(prototxt);
	pc->delete_operation(prototxt);
	pc->build_net_dag(prototxt);
	pc->init_operation(prototxt);
	pc->build_forward(prototxt);*/
	return 0;
}