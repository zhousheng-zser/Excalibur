#include "alcnn.hpp"
#include "../Excalibur/io.hpp"

namespace glasssix
{
	alcnn::alcnn(int device)
	{
		device_ = device;
		ipbbox = new ipbbox_net(device_);
		ipts = new ipts_net(device_);
		if (device_>=0)
		{
#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#else
			NO_GPU;
#endif
		}
	}

	alcnn::~alcnn()
	{
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#else
			NO_GPU;
#endif
		}
		delete ipbbox;
		delete ipts;
	}

#ifdef USE_OPENCV
	cv::Mat alcnn::safetycut(cv::Mat ori, cv::Rect rect)
	{
		cv::Mat mat(rect.height, rect.width, ori.type(), cv::Scalar(0, 0, 0));
		if (rect.x >= 0 && rect.y >= 0 && (rect.x + rect.width) <= ori.cols && (rect.y + rect.height) <= ori.rows)
		{
			ori(rect).copyTo(mat);
		}
		else
		{
			cv::Mat border_mat;
			int x_scale = std::max(std::max(-1 * rect.x, rect.x + rect.width - ori.cols), -1) + 1;
			int y_scale = std::max(std::max(-1 * rect.y, rect.y + rect.height - ori.rows), -1) + 1;
			cv::copyMakeBorder(ori, border_mat, y_scale, y_scale, x_scale, x_scale, cv::BORDER_CONSTANT);
			rect.x += x_scale;
			rect.y += y_scale;
			border_mat(rect).copyTo(mat);
		}
		return mat;
	}

	void alcnn::alignface_opencv(cv::Mat& img, cv::Mat& aligned)
	{
		cv::Mat tempMat;
		cv::resize(img, tempMat, cv::Size(60, 60), cv::INTER_CUBIC);
		std::shared_ptr<tensor> mat_tensor = nullptr;
		io::image2tensor(tempMat, mat_tensor);
		if (device_<0)
			ipbbox->Forward_cpu(mat_tensor);
		else
			ipbbox->Forward_native_gpu(mat_tensor, cublas_handle_);
		auto bbox_data = ipbbox->get_fc2()->cpu_data();
		auto headpose_data = ipbbox->get_fc3()->cpu_data();

		cv::Rect face_rect((int)(bbox_data[0] * img.cols), (int)(bbox_data[1] * img.rows), (int)((bbox_data[2] - bbox_data[0]) * img.cols), (int)((bbox_data[3] - bbox_data[1]) * img.rows));
		cv::Point2f center((bbox_data[0] + bbox_data[2]) * img.cols / 2, (bbox_data[1] + bbox_data[3]) * img.rows / 2);
		double angletheta = headpose_data[2] * 90;
		double arctheta = angletheta * CV_PI / 180;
		cv::Mat rot_mat = cv::getRotationMatrix2D(center, -1 * angletheta, 1.0);
		cv::Mat roted_img;
		cv::warpAffine(img, roted_img, rot_mat, img.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		int delta = (int)(sin(arctheta) * face_rect.height / 2);
		face_rect.x += delta;
		center.x += delta;
		int delta_pitch = (int)((1 - cos((double)headpose_data[1] * 90 * CV_PI / 180))*face_rect.height / 2);

		face_rect.y -= delta_pitch;
		face_rect.height += 2 * delta_pitch;

		int margin_height = (int)(face_rect.height * 1.2);
		int margin_x = (int)(center.x - margin_height / 2);
		int margin_y = (int)(center.y - margin_height / 2);

		cv::Rect margin_rect(margin_x, margin_y, margin_height, margin_height);
		cv::Mat C = safetycut(roted_img, margin_rect);
		//5IPTs
		cv::resize(C, tempMat, cv::Size(60, 60), cv::INTER_CUBIC);
		io::image2tensor(tempMat, mat_tensor);
		if (device_<0)
			ipts->Forward_cpu(mat_tensor);
		else
			ipts->Forward_native_gpu(mat_tensor, cublas_handle_);
		auto lmdk_data = ipts->get_fc2()->cpu_data();
		cv::Point2f center_eye((lmdk_data[0] + lmdk_data[2]) / 2 * C.cols, (lmdk_data[1] + lmdk_data[3]) / 2 * C.rows);
		cv::Point2f center_mouth((lmdk_data[6] + lmdk_data[8]) / 2 * C.cols, (lmdk_data[7] + lmdk_data[9]) / 2 * C.rows);
		cv::Point2f half_square(0.5f * C.cols, 0.25f * C.rows);

		float distance_x = center_eye.x - half_square.x;
		float distance_y = center_eye.y - half_square.y;
		float distance_me = (float)sqrt((center_mouth.x - center_eye.x) * (center_mouth.x - center_eye.x) + (center_mouth.y - center_eye.y) * (center_mouth.y - center_eye.y));
		float scale = distance_me / (C.rows * 0.5f);

		margin_rect.x += (int)distance_x;
		margin_rect.y += (int)distance_y;

		double tan = (lmdk_data[1] - lmdk_data[3]) / (lmdk_data[0] + lmdk_data[2]);
		double arctan = atan(tan) * 180 / CV_PI;

		cv::Mat rot_mat_2 = cv::getRotationMatrix2D(cv::Point2f(half_square.x + margin_rect.x, half_square.y + margin_rect.y), -3 * arctan, 1.0);
		cv::Mat roted_img_2;
		cv::warpAffine(roted_img, roted_img_2, rot_mat_2, roted_img.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		margin_rect.x = margin_rect.x + (int)((1 - scale) * margin_rect.width * 0.5f);
		margin_rect.y = margin_rect.y + (int)((1 - scale) * margin_rect.height * 0.25f);
		margin_rect.width = (int)(margin_rect.width * scale);
		margin_rect.height = (int)(margin_rect.height * scale);
		cv::Mat F = safetycut(roted_img_2, margin_rect);
		cv::Rect final_rect((int)(0.0f / 14 * F.cols), 0, (int)(14.0f / 14 * F.cols), F.rows);
		cv::Mat Gray(final_rect.height, final_rect.width, F.type(), cv::Scalar::all(0));
		cv::cvtColor(F(final_rect), Gray, CV_BGR2GRAY);
		cv::equalizeHist(Gray, Gray);
		cv::Mat colorimg;
		cv::merge(std::vector<cv::Mat>{ Gray, Gray, Gray }, colorimg);
		cv::resize(colorimg, aligned, cv::Size(128, 128), cv::INTER_CUBIC);
	}

	std::shared_ptr<tensor> alcnn::alignface(cv::Mat& img)
	{
		std::shared_ptr<tensor> mat_tensor = nullptr;
		cv::Mat aligned;
		alignface_opencv(img, aligned);
		io::image2tensor(aligned, mat_tensor);
		return mat_tensor;
	}

#endif

}