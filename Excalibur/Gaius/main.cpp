#include "ImageReader.hpp"
#include "ImagePyramid.hpp"
#include "FUSTDetector.hpp"
#include <filesystem>

using namespace excalibur;

int main()
{
	FUSTDtector FD = FUSTDtector();
	FD.LoadModel("D:\\Research\\MCL_Forward\\Excalibur\\model\\gaius_frontal.bin");
	cv::Mat mat = cv::imread("E:\\datasets\\LS3D-W\\300W-Testset-3D\\indoor_003.png", cv::IMREAD_GRAYSCALE);
	std::shared_ptr<ImagePyramid> img_pyramid = std::make_shared<ImagePyramid>(-1);
	std::shared_ptr<ImageTensor<unsigned char>> mat_data = nullptr;
	ImageReader::image2tensor(mat, mat_data);
	int32_t min_img_size = mat_data->height() <= mat_data->width() ? mat_data->height() : mat_data->width();
	min_img_size = (0 > 0 ?
		(min_img_size >= 0 ? 0 : min_img_size) :
		min_img_size);
	img_pyramid->SetImage1x(mat.data, mat.cols, mat.rows);
	img_pyramid->SetMinScale(40.f / min_img_size);
	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	auto res = FD.Detect(img_pyramid);
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "total detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000  << "ms" << std::endl << std::endl;
	for (int i = 0; i < res.size(); i++)
	{
		cv::rectangle(mat, cv::Rect(res[i].bbox.x, res[i].bbox.y, res[i].bbox.width, res[i].bbox.height), cv::Scalar(0, 0, 255), 2);
	}
	cv::imshow("propose", mat);
	cv::waitKey();
	/*float scale_factor = 0.0;
	std::shared_ptr<ImageTensor<unsigned char>> img_scaled = img_pyramid->GetNextScaleImage(&scale_factor);
	while (img_scaled != nullptr)
	{
		img_scaled = img_pyramid->GetNextScaleImage(&scale_factor);
		if (img_scaled!=nullptr)
		{
			cv::Mat imshow;
			ImageReader::tensor2image(img_scaled, imshow);
			cv::imshow("test", imshow);
			cv::waitKey();
			imshow.release();
		}
	}*/
	return 0;
}