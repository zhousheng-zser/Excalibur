#include "Excalibur_MathFunctions.hpp"
#include "ReLU_Layer.hpp"
#include <stdio.h>
#include <io.h>
#include <fstream>  
#include <iostream>
#include <fcntl.h>
using namespace Excalibur;


bool ReadProtoFromBinaryFile(const char* file, Message* net) {
	int fd = _open(file, O_RDONLY | O_BINARY);
	if (fd == -1) return false;

	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	bool success = net->ParseFromCodedStream(coded_input);
	delete coded_input;
	delete raw_input;
	_close(fd);
	return success;
}


void main() {

	int i = 0;
	double A[6] = { 1.0,2.0,1.0,-3.0,4.0,-1.0 };
	double B[6] = { 1.0,2.0,1.0,-3.0,4.0,-1.0 };
	double C[9] = { .5,.5,.5,.5,.5,.5,.5,.5,.5 };
	float * y = (float*)malloc(6 * sizeof(float));
	float * a = (float*)malloc(6 * sizeof(float));
	float * b = (float*)malloc(6 * sizeof(float));
	float * c = (float*)malloc(9 * sizeof(float));
	for (int i = 0; i < 6; i++)
	{
		a[i] = i;
		b[i] = 10 - i;
	}
	for (int i = 0; i < 9; i++)
	{
		c[i] = 64 * i % 255;
	}
	int M = 3; // row of A and C
	int N = 3; // col of B and C
	int K = 2; // col of A and row of B

	double alpha = 1.0;
	double beta = 0.0;
	//LOG(INFO) << "TEST";
	Excalibur_MathFunctions exmath = Excalibur_MathFunctions();
	exmath.excalibur_cpu_gemm(CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, B, beta, C);
	vsMul(6, a, b, y);
	Excalibur_MathFunctions exmath1 = Excalibur_MathFunctions(0, GPU);
	Excalibur_MathFunctions exmath2 = Excalibur_MathFunctions(1, GPU);
	caffe::NetParameter net;
	bool success = ReadProtoFromBinaryFile("det1.caffemodel", &net);
	caffe::LayerParameter& param = *net.mutable_layer(4);
	std::vector<Pandora_Blob<float>*> bottom_data(1);
	std::vector<Pandora_Blob<float>*> top_data(1);
	bottom_data[0] = new Pandora_Blob<float>(std::vector<int>{270, 1, 1, 1}, 1, GPU);
	top_data[0] = new Pandora_Blob<float>(std::vector<int>{270, 1, 1, 1}, 1, GPU);
	float* temp = bottom_data[0]->mutable_cpu_data();
	for (int i = 0; i < 270; i++)
	{
		temp[i] = param.blobs(0).data(i);
	}
	ReLU_Layer<float> relu_layer = ReLU_Layer<float>(param, 1, GPU);
	relu_layer.Forward_cpu(bottom_data, top_data);
	for (int i = 0; i < 270; i++) {
		std::cout << (top_data[0]->cpu_data())[i]<< " ";
	}
}