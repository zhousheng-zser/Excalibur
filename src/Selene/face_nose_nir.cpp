#include "../../include/Selene/face_nose_nir.hpp"
#include "../../include/Selene/face_nir_net.hpp"
#include "../../include/Selene/nose_nir_net.hpp"
#include <glasssix/tensor.hpp>
#include <glasssix/syncedmem.hpp>

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		Face_nose_nir::Face_nose_nir(int device)
		{
			face_nir_net_.reset(new Face_nir_net(device));
			nose_nir_net_.reset(new Nose_nir_net(device));
		}



		Face_nose_nir::~Face_nose_nir() {}



		/// <summary>
		/// get area of face and nose
		/// </summary>
		/// <param name="image_nir">near-infrared image data</param>
		/// <param name="bbox">detected humanface bboxes</param>
		/// <param name="landmarks">detected humanface landmarks</param>
		/// <param name="face_nir">face area in near-infrared image</param>
		/// <param name="nose_nir">nose area in near-infrared image</param>
		void Face_nose_nir::face_nose_area(const std::shared_ptr<tensor<unsigned char>> &image_nir, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, 
			std::vector<std::shared_ptr<tensor<unsigned char>>> &face_nir, std::shared_ptr<tensor<unsigned char>> &nose_nir)
		{
			CHECK_EQ(bbox.size(), 1);
			CHECK_EQ(landmarks.size(), 1);
			CHECK_EQ(landmarks[0].size() / 2, 5);

			std::shared_ptr<tensor<unsigned char>> ROI, rotated_ROI, face_mat, nose_mat;
			glasssix::excalibur::rectangle<int> MarginRect = glasssix::excalibur::rectangle<int>(bbox[0][0] - bbox[0][3] * 0.2,
				bbox[0][1] - bbox[0][2] * 0.2,
				bbox[0][3] * 1.4f,
				bbox[0][2] * 1.4f);

			tensor_operation_cpu::safty_cut_cpu(image_nir, ROI, &MarginRect);

			point<float> ldmk5[5];
			for (size_t j = 0; j < landmarks[0].size() / 2; j++)
			{
				ldmk5[j] = point<float>(landmarks[0][2 * j] - MarginRect.x, landmarks[0][2 * j + 1] - MarginRect.y);
			}
			point<float> center_eye = point<float>((ldmk5[0].x + ldmk5[1].x) / 2, (ldmk5[0].y + ldmk5[1].y) / 2);
			point<float> center_mouth = point<float>((ldmk5[3].x + ldmk5[4].x) / 2, (ldmk5[3].y + ldmk5[4].y) / 2);
			point<float> center = point<float>((center_eye.x + center_mouth.x) / 2, (center_eye.y + center_mouth.y) / 2);
			double tan = (center_eye.x - center_mouth.x) / (center_eye.y - center_mouth.y);
			double arctan = atan(tan) * 180 / 3.1415926;
			tensor_operation_cpu::rotate_with_points_cpu(ROI, rotated_ROI, center, -1 * arctan);

			//nose_area
			double mouse_distance = sqrt((ldmk5[3].x - ldmk5[4].x) * (ldmk5[3].x - ldmk5[4].x) + (ldmk5[3].y - ldmk5[4].y) * (ldmk5[3].y - ldmk5[4].y));
			glasssix::excalibur::rectangle<int> nose_rect = glasssix::excalibur::rectangle<int>(ldmk5[2].x - 0.5 * mouse_distance,
				ldmk5[2].y - 0.55 * mouse_distance,
				mouse_distance, mouse_distance);
			tensor_operation_cpu::safty_cut_cpu(rotated_ROI, nose_mat, &nose_rect);
			tensor_operation_cpu::resize_cpu(nose_mat, nose_nir, 48, 48);

			//face area
			double distance = sqrt((center_eye.x - center_mouth.x) * (center_eye.x - center_mouth.x) + (center_eye.y - center_mouth.y) * (center_eye.y - center_mouth.y));
			double cos = (center_mouth.y - center_eye.y) / distance;
			double sin = (center_mouth.x - center_eye.x) / distance;
			point<float> new_center_eye = point<float>(center_eye.x + (float)(sin * distance / 2), (float)(center_eye.y - (1 - cos) * distance / 2));
			point<float> new_center_mouth = point<float>(center_mouth.x - (float)(sin * distance / 2), (float)(center_mouth.y + (1 - cos) * distance / 2));

			{
				int x_diff = 0;
				int y_diff = 0;
				std::shared_ptr<tensor<unsigned char>> temp;

				glasssix::excalibur::rectangle<float> face_rect = glasssix::excalibur::rectangle<float>(new_center_eye.x - distance + x_diff,
					new_center_eye.y - distance / 2 + y_diff,
					distance * 2, distance * 2);
				tensor_operation_cpu::safty_cut_cpu(rotated_ROI, face_mat, &face_rect);
				tensor_operation_cpu::resize_cpu(face_mat, temp, 50, 50);
				tensor_operation_cpu::lbp_feature_cpu(temp, temp);
				face_nir.push_back(temp);
			}
		}



		/// <summary>
		/// detect on near-infrared image, whether color photo detected, return true if pass(not color photo)
		/// </summary>
		/// <param name="nir_color_image">near-infrared image data</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="bbox">detected humanface bboxes</param>
		/// <param name="landmarks">detected humanface landmarks</param>
		/// <param name="thresh">thresh[0]: threshold value of face-judge-model, 0.9 by default; thresh[1]: threshold value of nose-judge-model, 0.85 by default</param>
		/// <param name="value">value[0]: return value of face-judge-model; value[1]: return value of nose-judge-model</param>
		/// <param name="order">order type of near-infrared image: NCHW(0) / NHWC(1)</param>
		bool Face_nose_nir::judge(const unsigned char* nir_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order)
		{
			std::shared_ptr<tensor<unsigned char>> image_nir, gray_nir;
			if (order == 0)
			{
				image_nir.reset(new tensor<unsigned char>(std::vector<int>{1, 3, height, width}, -1, NCHW));
			}
			else if (order == 1)
			{
				image_nir.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, 3}, -1, NHWC));
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			memcpy(image_nir->mutable_cpu_data(), nir_color_image, 1 * 3 * height * width * sizeof(unsigned char));

			tensor_operation_cpu::rgb2gray_cpu(image_nir, gray_nir);
			tensor_operation_cpu::gaussian_blur_cpu(gray_nir, gray_nir);

			std::vector<std::shared_ptr<tensor<unsigned char>>> face_nir;
			std::shared_ptr<tensor<unsigned char>> nose_nir;
			face_nose_area(gray_nir, bbox, landmarks, face_nir, nose_nir);

			face_nir_net_->Forward(face_nir[0]);
			const float* face_prob_data = face_nir_net_->get_conv6_1()->cpu_data();
			value[0] = face_prob_data[0];
			bool face_judge_nir = value[0] > thresh[0] ? true : false;

			nose_nir_net_->Forward(nose_nir);
			const float* nose_prob_data = nose_nir_net_->get_conv6_1()->cpu_data();
			value[1] = nose_prob_data[0];
			bool nose_judge_nir = value[1] > thresh[1] ? true : false;

			if (face_judge_nir && nose_judge_nir)
			{
				return true;//real human detected
			}
			else
			{
				return false;//color photo detected
			}
		}
	}
}