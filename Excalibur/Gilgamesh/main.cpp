#include "../Excalibur/tensoroperation.hpp"
#include <glasssix/timer.hpp>
using namespace excalibur;
int main()
{
	glasssix::Timer timer;
	cv::Mat mat = cv::imread("C:\\Users\\BALTHASAR\\Desktop\\0.jpg");
	std::shared_ptr<excalibur::tensor<unsigned char>> tensor_mat;
	excalibur::tensoroperation::convert2tensor(mat, tensor_mat);
	std::shared_ptr<excalibur::tensor<unsigned char>> tensor_mat_gray;
	//tensoroperation::rgb2gray_cpu(tensor_mat, tensor_mat_gray);
	cv::Mat resize_mat;
	timer.Start();
	for (size_t i = 0; i < 100; i++)
	{
		//tensoroperation::resize_cpu(tensor_mat, tensor_mat_gray, 100, 100, 0);
		cv::resize(mat, resize_mat, cv::Size(100, 100), 0, 0, 0);
	}
	
	timer.Stop();
	std::cout << timer.GetElapsedMilliseconds() / 100;
	cv::Mat show_mat;
	/*excalibur::tensoroperation::convert2mat(tensor_mat_gray, show_mat);
	cv::imshow("test", show_mat);*/
	cv::waitKey();
	return 0;
}