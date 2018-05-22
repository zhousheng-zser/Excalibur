#include "../Excalibur/tensoroperation.hpp"

using namespace excalibur;
int main()
{
	cv::Mat mat = cv::imread("C:\\Users\\BALTHASAR\\Desktop\\0.jpg");
	std::shared_ptr<excalibur::tensor<unsigned char>> tensor_mat;
	excalibur::tensoroperation::convert2tensor(mat, tensor_mat);
	std::shared_ptr<excalibur::tensor<unsigned char>> tensor_mat_gray;
	//tensoroperation::rgb2gray_cpu(tensor_mat, tensor_mat_gray);
	tensoroperation::bilinear_resize_cpu(tensor_mat, tensor_mat_gray, 100, 100);
	
	cv::Mat show_mat;
	excalibur::tensoroperation::convert2mat(tensor_mat_gray, show_mat);
	cv::imshow("test", show_mat);
	cv::waitKey();
	return 0;
}