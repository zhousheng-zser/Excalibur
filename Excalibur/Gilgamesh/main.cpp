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
	for (size_t i = 0; i < 1; i++)
	{
		tensoroperation::resize_cpu(tensor_mat, tensor_mat_gray, 600, 600, tensoroperation::Nearest);
		//cv::resize(mat, resize_mat, cv::Size(600, 600), 0, 0, 1);
		//tensoroperation::copy_make_border_cpu(tensor_mat, tensor_mat_gray, 40, 23, 64, 71, 0, (unsigned char)128);
		/*tensoroperation::rotate_cpu(tensor_mat, tensor_mat_gray, 3.14 / 6, 
			125, 297, tensoroperation::Nearest, (unsigned char)128);*/
		//tensoroperation::rgb2gray_cpu(tensor_mat, tensor_mat_gray);
		rectangle<int> rect(120, 125, 350, 350);
		tensoroperation::draw_rectangle_cpu(tensor_mat_gray, rect, 2, color(255, 255, 0));
	}
	
	timer.Stop();
	std::cout << timer.GetElapsedMilliseconds() / 1;
	cv::Mat show_mat;
	excalibur::tensoroperation::convert2mat(tensor_mat_gray, show_mat);
	cv::imshow("test", show_mat);
	cv::waitKey();
	return 0;
}