#include "alignment.hpp"
#include "../Excalibur/io.hpp"

namespace glasssix
{
	alignment::alignment(int device)
	{
		device_ = device;
		ipbbox.reset(new ipbbox_net(device_));
		ipts.reset(new ipts_net(device_));
		if (device_ >= 0)
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

	alignment::~alignment()
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
	}

	cv::Mat alignment::saftycut(cv::Mat ori_image, cv::Rect roi)
	{
		cv::Mat output = cv::Mat(roi.height, roi.width, ori_image.type());
		if (roi.x>=0&&roi.y>=0&&(roi.x + roi.width <= ori_image.cols)&&(roi.y+roi.height <= ori_image.rows))
		{
			ori_image(roi).copyTo(output);
		}
		else
		{
			int top = std::max(0, -1 * roi.x);
			int bottom = std::max(roi.y + roi.height - ori_image.rows, 0);
			int left = std::max(0, -1 * roi.y);
			int right = std::max(roi.x + roi.width - ori_image.cols, 0);
			cv::Mat temp_origin_with_border;
			cv::copyMakeBorder(ori_image, temp_origin_with_border, top, bottom, left, right, cv::BORDER_CONSTANT);
			roi.x += left;
			roi.y += top;
			temp_origin_with_border(roi).copyTo(output);
		}
		return output;
	}


	void alignment::alignment_face(cv::Mat& img, cv::Mat& aligned_face)
	{
		cv::Mat ipbbox_input_mat;
		cv::resize(img, ipbbox_input_mat, cv::Size(60, 60), cv::INTER_CUBIC);
		io::image2tensor(ipbbox_input_mat, tensor_data);
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			ipbbox->Forward_native_gpu(tensor_data, cublas_handle_);
#else
			NO_GPU;
#endif
		}
		else
		{
			ipbbox->Forward_cpu(tensor_data);
		}
		const float* ipbbox_fc2 = ipbbox->get_fc2()->cpu_data();
		const float* ipbbox_fc3 = ipbbox->get_fc3()->cpu_data();

		cv::Rect face_rect((int)(ipbbox_fc2[0] * img.cols), (int)(ipbbox_fc2[1] * img.rows), 
			(int)((ipbbox_fc2[2] - ipbbox_fc2[0]) * img.cols), (int)((ipbbox_fc2[3] - ipbbox_fc2[1]) * img.rows));

		cv::Point2f center((ipbbox_fc2[0] + ipbbox_fc2[2]) * img.cols / 2, (ipbbox_fc2[1] + ipbbox_fc2[3]) * img.rows / 2);

		double angletheta = ipbbox_fc3[2] * 90; // roll data
		double arctheta = angletheta * CV_PI / 180;

		cv::Mat rot_mat = cv::getRotationMatrix2D(center, -1 * angletheta, 1.0);
		cv::Mat roted_img;
		cv::warpAffine(img, roted_img, rot_mat, img.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		int delta = (int)(sin(arctheta) * face_rect.height / 2);
		face_rect.x += delta;
		center.x += delta;
		int delta_pitch = (int)((1 - cos((double)ipbbox_fc3[1] * 90 * CV_PI / 180))*face_rect.height / 2);

		face_rect.y -= delta_pitch;
		face_rect.height += 2 * delta_pitch;

		int margin_height = (int)(face_rect.height * 1.2);
		int margin_x = (int)(center.x - margin_height / 2);
		int margin_y = (int)(center.y - margin_height / 2);

		cv::Rect margin_rect(margin_x, margin_y, margin_height, margin_height);

		cv::Mat C = saftycut(roted_img, margin_rect);
		//5IPTs
		cv::Mat ipts_input_mat;
		cv::resize(C, ipts_input_mat, cv::Size(60, 60), cv::INTER_CUBIC);
		io::image2tensor(ipts_input_mat, tensor_data);
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			ipts->Forward_native_gpu(tensor_data, cublas_handle_);
#else
			NO_GPU;
#endif
		}
		else
		{
			ipts->Forward_cpu(tensor_data);
		}
		const float* ipts_fc2 = ipts->get_fc2()->cpu_data();
		cv::Point2f center_eye((ipts_fc2[0] + ipts_fc2[2]) / 2 * C.cols, (ipts_fc2[1] + ipts_fc2[3]) / 2 * C.rows);
		cv::Point2f center_mouth((ipts_fc2[6] + ipts_fc2[8]) / 2 * C.cols, (ipts_fc2[7] + ipts_fc2[9]) / 2 * C.rows);
		cv::Point2f half_quarter(0.5f * C.cols, 0.25f * C.rows);

		float distance_x = center_eye.x - half_quarter.x;
		float distance_y = center_eye.y - half_quarter.y;
		float distance_me = (float)sqrt((center_mouth.x - center_eye.x) * (center_mouth.x - center_eye.x) + (center_mouth.y - center_eye.y) * (center_mouth.y - center_eye.y));
		float scale = distance_me / (C.rows * 0.5f);

		margin_rect.x += (int)distance_x;
		margin_rect.y += (int)distance_y;

		double tan = (ipts_fc2[1] - ipts_fc2[3]) / (ipts_fc2[0] + ipts_fc2[2]);
		double arctan = atan(tan) * 180 / CV_PI;

		cv::Mat rot_mat_2 = cv::getRotationMatrix2D(cv::Point2f(half_quarter.x + margin_rect.x, half_quarter.y + margin_rect.y), -3 * arctan, 1.0);
		cv::Mat roted_img_2;
		cv::warpAffine(roted_img, roted_img_2, rot_mat_2, roted_img.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		margin_rect.x = margin_rect.x + (int)((1 - scale) * margin_rect.width * 0.5f);
		margin_rect.y = margin_rect.y + (int)((1 - scale) * margin_rect.height * 0.25f);
		margin_rect.width = (int)(margin_rect.width * scale);
		margin_rect.height = (int)(margin_rect.height * scale);

		cv::Mat F = saftycut(roted_img_2, margin_rect);
		cv::Rect final_rect((int)(0.0f / 14 * F.cols), 0, (int)(14.0f / 14 * F.cols), F.rows);
		cv::Mat Gray(final_rect.height, final_rect.width, F.type(), cv::Scalar::all(0));
		cv::cvtColor(F(final_rect), Gray, CV_BGR2GRAY);
		cv::equalizeHist(Gray, Gray);

		cv::Mat colorimg;
		cv::merge(std::vector<cv::Mat>{ Gray, Gray, Gray }, colorimg);
		cv::resize(colorimg, aligned_face, cv::Size(128, 128), cv::INTER_CUBIC);
	}
}
