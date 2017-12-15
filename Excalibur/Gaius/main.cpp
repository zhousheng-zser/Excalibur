#include "ImageReader.hpp"
#include "ImagePyramid.hpp"
#include "FUSTDetector.hpp"
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
	float scale_factor = 0.0;
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
	}
	return 0;
}