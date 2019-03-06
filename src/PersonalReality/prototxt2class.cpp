#include "prototxt2class.hpp"



prototxt2class::prototxt2class(std::string outpath, std::string netname)
{
	cpp = std::ofstream(outpath + "\\" + netname + ".cpp", std::ios::app);
	hpp = std::ofstream(outpath + "\\" + netname + ".hpp", std::ios::app);
}


prototxt2class::~prototxt2class()
{
}

void prototxt2class::declear_params(std::string caffemodel, std::string netname)
{
	NetParameter param;
	ReadProtoFromBinaryFile(caffemodel.c_str(), &param);
	for (int i = 1; i < param.layer_size(); i++)
	{
		LayerParameter& layer_param = *param.mutable_layer(i);
		int n = param.mutable_layer(i)->mutable_blobs()->size();
		if (n)
		{
			std::string layer_name = layer_param.name();
			std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);
			hpp << "Declear_Params(" + layer_name + "_weights);" << std::endl;
			cpp << "Copy_Params(" + layer_name + "_weights, " + netname + ");" << std::endl;
			if (n>1)
			{
				hpp << "Declear_Params(" + layer_name + "_bias);" << std::endl;
				cpp << "Copy_Params(" + layer_name + "_bias, " + netname + ");" << std::endl;
			}
			std::cout << i << " " << layer_name << std::endl;
		}
		std::cout << i  << std::endl;
	}
	hpp << std::endl;
	cpp << std::endl;
}

void prototxt2class::declear_operation_neuron(std::string prorotxt)
{
	NetParameter param;
	ReadProtoFromTextFile(prorotxt.c_str(), &param);
	for (int i = 1; i < param.layer_size(); i++)
	{
		LayerParameter& layer_param = *param.mutable_layer(i);
		std::string type_name = layer_param.type();
		std::string layer_name = layer_param.name();
		std::transform(type_name.begin(), type_name.end(), type_name.begin(), tolower);
		std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);
		hpp << "Declear_Opration(" + type_name + ", " + layer_name + ");" << std::endl;
		if (type_name!="prelu"||type_name!="relu")
		{
			hpp << "Neuron_Name(" + layer_name + ");" << std::endl;
		}
	}
	hpp  << std::endl;
}


void prototxt2class::delete_operation(std::string prorotxt)
{
	NetParameter param;
	ReadProtoFromTextFile(prorotxt.c_str(), &param);
	for (int i = 1; i < param.layer_size(); i++)
	{
		LayerParameter& layer_param = *param.mutable_layer(i);
		std::string layer_name = layer_param.name();
		std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);
		hpp << "delete " + layer_name + ";" << std::endl;
	}
}

void prototxt2class::build_net_dag(std::string prorotxt)
{
	NetParameter param;
	ReadProtoFromTextFile(prorotxt.c_str(), &param);
	for (int i = 1; i < param.layer_size(); i++)
	{
		LayerParameter& layer_param = *param.mutable_layer(i);
		std::string layer_name = layer_param.name();
		std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);
		net_dag.insert({ i, layer_name });
	}
}


void prototxt2class::init_operation(std::string prorotxt)
{
	NetParameter param;
	ReadProtoFromTextFile(prorotxt.c_str(), &param);
	for (auto elem : net_dag)
	{
		int index = elem.first;
		LayerParameter& layer_param = *param.mutable_layer(index);
		std::string type_name = layer_param.type();
		std::transform(type_name.begin(), type_name.end(), type_name.begin(), tolower);
		if (type_name=="convolution")
		{
			bool bias_term = layer_param.convolution_param().bias_term();
			int num_output = layer_param.convolution_param().num_output();
			int kernel_size = layer_param.convolution_param().kernel_size(0);
			int stride = layer_param.convolution_param().stride_size();
			int pad = layer_param.convolution_param().pad_size();
			int num_input = 666;
			if (bias_term)
			{
				cpp << "Init_Conv_Params(" << elem.second + ", " <<
					num_input << ", " << num_output << ", " <<
					kernel_size << ", " << stride << ", " <<
					pad << ", true);" << std::endl;
			}
			else
			{
				cpp << "Init_Conv_Params(" << elem.second + ", " <<
					num_input << ", " << num_output << ", " <<
					kernel_size << ", " << stride << ", " <<
					pad << ", false);" << std::endl;
			}
		}
		if (type_name == "relu")
		{
			int num_input = 666;
			cpp << "Init_PReLU_Params(" + elem.second << ", " << num_input << ", true);" << std::endl;
		}
		if (type_name == "prelu")
		{
			int num_input = 666;
			cpp << "Init_PReLU_Params(" + elem.second << ", " << num_input << ", false);" << std::endl;
		}
		if (type_name=="pooling")
		{
			PoolingParameter_PoolMethod poolmethod = layer_param.pooling_param().pool();
			int kernel_size = layer_param.pooling_param().kernel_size();
			int pad = layer_param.pooling_param().pad();
			int stride = layer_param.pooling_param().stride();
			if (poolmethod == PoolingParameter_PoolMethod_MAX)
			{
				cpp << "Init_Pooling_Params(" + elem.second << ", " << kernel_size << ", " << stride << ", " << pad << ", 0);" << std::endl;
			}
			if (poolmethod == PoolingParameter_PoolMethod_AVE)
			{
				cpp << "Init_Pooling_Params(" + elem.second << ", " << kernel_size << ", " << stride << ", " << pad << ", 1);" << std::endl;
			}
		}
		if (type_name == "eltwise")
		{
			EltwiseParameter_EltwiseOp eltmethod = layer_param.eltwise_param().operation();
			if (eltmethod == EltwiseParameter_EltwiseOp_SUM)
			{
				cpp << "Init_Eltwise_Params(" << elem.second << ", 0);" << std::endl;
			}
			if (eltmethod == EltwiseParameter_EltwiseOp_MAX)
			{
				cpp << "Init_Eltwise_Params(" << elem.second << ", 1);" << std::endl;
			}
		}
		if (type_name == "innerproduct")
		{
			int num_output = layer_param.inner_product_param().num_output();
			bool bias_term = layer_param.inner_product_param().bias_term();
			if (bias_term)
			{
				cpp << "Init_InnerProduct_Params(" << elem.second << ", 666, 66, 6, " << num_output << ", true);" << std::endl;
			}
			else
			{
				cpp << "Init_InnerProduct_Params(" << elem.second << ", 666, 66, 6, " << num_output << ", false);" << std::endl;
			}
		}
		if (type_name=="softmax")
		{
			cpp << "Init_Softmax_Params(" << elem.second << ", 666);" << std::endl;
		}
	}
	cpp << std::endl;
}


void prototxt2class::build_forward(std::string prorotxt)
{
	NetParameter param;
	ReadProtoFromTextFile(prorotxt.c_str(), &param);
	for (auto elem : net_dag)
	{
		int index = elem.first;
		LayerParameter& layer_param = *param.mutable_layer(index);
		std::string type_name = layer_param.type();
		std::transform(type_name.begin(), type_name.end(), type_name.begin(), tolower);
		int bottom_size = layer_param.bottom_size();
		std::vector<std::string> bottom_names;
		for (int i = 0; i < bottom_size; i++)
		{
			bottom_names.push_back(layer_param.bottom(i));
		}
		if (bottom_names.size()<2)
		{
			if (type_name=="relu"|| type_name=="prelu")
			{
				cpp << elem.second << "->Forward_cpu(" << bottom_names[0] << "_top_data);" << std::endl;
			}
			else
			{
				cpp << elem.second << "->Forward_cpu(" << bottom_names[0] << "_top_data, " << elem.second << "_top_data);" << std::endl;
			}
		}
		else
		{
			cpp << elem.second << "->Forward_cpu(std::vector<std::shared_ptr<tensor>>{" << bottom_names[0] << "_top_data, "<<
				bottom_names[1] << "_top_data}, " << elem.second << "_top_data);" << std::endl;
		}
	}
	cpp << std::endl;
}
