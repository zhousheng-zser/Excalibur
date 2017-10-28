//#include "datafile.hpp"
#include "prototxt2class.hpp"
int main()
{
	/*datafile* df = new datafile("D:\\Research\\fastfacedetection\\model\\mtcnn_onet_data.hpp");
	df->writedatahpp("D:\\Research\\fastfacedetection\\model\\det3-half.caffemodel", "ONet", "fastface");*/
	/*NetParameter param;
	ReadProtoFromTextFile("E:\\rec-bench\\model\\suffiver_deploy.prototxt", &param);
	for (int i = 1; i < param.layer_size(); i++)
	{
		LayerParameter& layer_param = *param.mutable_layer(i);
		std::cout << layer_param.name() << ", " << layer_param.bottom(0) << std::endl;
	}*/
	prototxt2class* pc = new prototxt2class("D:\\Research\\fastfacedetection\\model", "ONet");
	std::string prototxt = "D:\\Research\\fastfacedetection\\model\\det3-half_memory.prototxt";
	pc->declear_params("D:\\Research\\fastfacedetection\\model\\det3-half.caffemodel", "ONet");
	pc->declear_operation_neuron(prototxt);
	pc->delete_operation(prototxt);
	pc->build_net_dag(prototxt);
	pc->init_operation(prototxt);
	pc->build_forward(prototxt);
	return 0;
}