#include "datafile.hpp"
#include "prototxt2class.hpp"
int main()
{
	datafile* df = new datafile("C:\\DLFramework\\MCL_Forward\\Excalibur\\fastfacedetection\\mtcnn_onet_data_half.hpp");
	df->writedatahpp("C:\\DLFramework\\MCL_Forward\\Excalibur\\fastfacedetection\\det3-half.caffemodel", "ONet", "fastface");
	
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