#include <algorithm>
#include <vector>
#include "Diodorus.hpp"

using namespace glasssix::dionysios;
using namespace glasssix::longinus;
using namespace glasssix::excalibur;

Diodorus::Diodorus(int device): device_(device)
{
#ifdef USE_OPENCV
	svm_ = cv::ml::SVM::create();
	svm_ = cv::ml::SVM::load("D:/projects/data/AntiSpoofing/crop/hog_features.xml");
#endif

	saturate_data_ = (unsigned char*)malloc(168 * 128 * sizeof(unsigned char));
	face_sobel_data_ = (unsigned char*)malloc(168 * 128 * sizeof(unsigned char));
}
Diodorus::~Diodorus()
{
	delete saturate_data_;
	delete face_sobel_data_;
}

void LBPP8R1(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst)
{
	CHECK_EQ(src->num(), 1);
	CHECK_EQ(src->channels(), 1);
	int width = src->width();
	int height = src->height();

	dst.reset(new tensor<unsigned char>(std::vector<int>{1, 1, height - 2, width - 2}, -1, NCHW));
	const unsigned char* src_data = src->cpu_data();
	unsigned char* dst_data = dst->mutable_cpu_data();

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			unsigned char value = 0;
			int pos = 0;

			unsigned char u = src_data[i * width + j];
			if (src_data[(i - 1) * width + (j - 1)] > u)
			{
				value += 1 << pos;
			}

			pos++;

			if (src_data[(i - 1) * width + j] > u)
			{
				value += 1 << pos;
			}

			pos++;

			if (src_data[(i - 1) * width + (j + 1)] > u)
			{
				value += 1 << pos;
			}

			pos++;

			if (src_data[i * width + (j + 1)] > u)
			{
				value += 1 << pos;
			}

			pos++;

			if (src_data[(i + 1) * width + (j + 1)] > u)
			{
				value += 1 << pos;
			}

			pos++;

			if (src_data[(i + 1) * width + j] > u)
			{
				value += 1 << pos;
			}

			pos++;

			if (src_data[(i + 1) * width + (j - 1)] > u)
			{
				value += 1 << pos;
			}

			if (src_data[i * width + (j - 1)] > u)
			{
				value += 1 << pos;
			}

			std::vector<unsigned char> circle_values;
			circle_values.push_back(value);
			for (int i = 0; i < 7; i++)
			{
				unsigned char temp = value << 1;
				if (value >= 128)
				{
					temp++;//add 1 when highest bit is 1				
				}
				circle_values.push_back(temp);
				value = temp;
			}

			//if ((circle_values[0] == 240) || (circle_values[0] == 120) || (circle_values[0] == 60) || (circle_values[0] == 30) || 
			//	(circle_values[0] == 15) || (circle_values[0] == 135) || (circle_values[0] == 195) || (circle_values[0] == 225))
			//{
			//	std::cout << int(circle_values[0]) << "," << int(circle_values[1]) << "," << int(circle_values[2]) << "," << int(circle_values[3]) << ","
			//            << int(circle_values[4]) << "," << int(circle_values[5]) << "," << int(circle_values[6]) << "," << int(circle_values[7]) << std::endl;
			//}

			std::sort(circle_values.begin(), circle_values.end());
			dst_data[(i - 1) * (width - 2) + (j - 1)] = circle_values[0];
		}
	}
}

std::shared_ptr<glasssix::excalibur::tensor<unsigned char>> Diodorus::getFaceArea(const unsigned char* origine, int channels, int height, int width,
	std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks, int order)
{
	CHECK_EQ(landmarks.size(), 1);
	CHECK_EQ(bbox.size(), 1);

	std::shared_ptr<tensor<unsigned char>> ori_image, ROI, rotated_ROI, final_mat, res;

	if (order == 0)
	{
		ori_image.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_, NCHW));
	}
	else
	{
		ori_image.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, device_, NHWC));
	}

	if (device_ < 0)
	{
		memcpy(ori_image->mutable_cpu_data(), origine, 1 * channels * height * width * sizeof(unsigned char));

		CHECK_EQ(landmarks[0].size() / 2, 5);
		glasssix::excalibur::rectangle<int> MarginRect = glasssix::excalibur::rectangle<int>(bbox[0][0] - bbox[0][3] * 0.2,
			bbox[0][1] - bbox[0][2] * 0.2,
			bbox[0][3] * 1.4f,
			bbox[0][2] * 1.4f);

		tensor_operation_cpu::safty_cut_cpu(ori_image, ROI, &MarginRect);

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

		double distance = sqrt((center_eye.x - center_mouth.x) * (center_eye.x - center_mouth.x) + (center_eye.y - center_mouth.y) * (center_eye.y - center_mouth.y));
		double cos = (center_mouth.y - center_eye.y) / distance;
		double sin = (center_mouth.x - center_eye.x) / distance;
		point<float> new_center_eye = point<float>(center_eye.x + (float)(sin * distance / 2), (float)(center_eye.y - (1 - cos) * distance / 2));
		point<float> new_center_mouth = point<float>(center_mouth.x - (float)(sin * distance / 2), (float)(center_mouth.y + (1 - cos) * distance / 2));

		float rect_w = distance * 1.4;
		glasssix::excalibur::rectangle<float> final_rect = glasssix::excalibur::rectangle<float>(new_center_eye.x - 0.6 * distance,
			new_center_eye.y - 0.5 * distance,
			rect_w * 1.3125, rect_w);

		tensor_operation_cpu::safty_cut_cpu(rotated_ROI, final_mat, &final_rect);
		tensor_operation_cpu::resize_cpu(final_mat, res, 168, 128);
	}
	else
	{
		cudaMemcpy(ori_image->mutable_gpu_data(), origine, 1 * channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);

		CHECK_EQ(landmarks[0].size() / 2, 5);
		glasssix::excalibur::rectangle<int> MarginRect = glasssix::excalibur::rectangle<int>(bbox[0][0] - bbox[0][3] * 0.2,
			bbox[0][1] - bbox[0][2] * 0.2,
			bbox[0][3] * 1.4f,
			bbox[0][2] * 1.4f);

		tensor_operation_gpu::safty_cut_gpu(ori_image, ROI, &MarginRect);

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

		tensor_operation_gpu::rotate_with_points_gpu(ROI, rotated_ROI, center, -1 * arctan);

		double distance = sqrt((center_eye.x - center_mouth.x) * (center_eye.x - center_mouth.x) + (center_eye.y - center_mouth.y) * (center_eye.y - center_mouth.y));
		double cos = (center_mouth.y - center_eye.y) / distance;
		double sin = (center_mouth.x - center_eye.x) / distance;
		point<float> new_center_eye = point<float>(center_eye.x + (float)(sin * distance / 2), (float)(center_eye.y - (1 - cos) * distance / 2));
		point<float> new_center_mouth = point<float>(center_mouth.x - (float)(sin * distance / 2), (float)(center_mouth.y + (1 - cos) * distance / 2));

		float rect_w = distance * 1.4;
		glasssix::excalibur::rectangle<float> final_rect = glasssix::excalibur::rectangle<float>(new_center_eye.x - 0.6 * distance,
			new_center_eye.y - 0.5 * distance,
			rect_w * 1.3125, rect_w);

		tensor_operation_gpu::safty_cut_gpu(rotated_ROI, final_mat, &final_rect);
		tensor_operation_gpu::resize_gpu(final_mat, res, 168, 128);
	}

	return res;
}

bool Diodorus::aliveDetect(const unsigned char* srcColorVSL, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoVSL,
	int channels, int height, int width, int order,
	const unsigned char* srcColorNIR, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoNIR)
{
	CHECK_GT(face_infoVSL.size(), 0);

	std::vector<std::vector<int>> bboxMaxNIR, bboxMaxVSL;
	std::vector<std::vector<int>> landmarksMaxNIR, landmarksMaxVSL;
	glasssix::longinus::extract_biggest_faceinfo(face_infoVSL, bboxMaxVSL, landmarksMaxVSL);
	CHECK_EQ(bboxMaxVSL.size(), 1);
	CHECK_EQ(landmarksMaxVSL.size(), 1);

	if (!face_infoNIR.empty())
	{
		//detect on visible light and near infrared simultaneously
		longinus::extract_biggest_faceinfo(face_infoNIR, bboxMaxNIR, landmarksMaxNIR);
		CHECK_EQ(bboxMaxNIR.size(), 1);
		CHECK_EQ(landmarksMaxNIR.size(), 1);
	}
	else
	{
		//detect on visible light only
		float xLeftEye = landmarksMaxVSL[0][0];
		float yLeftEye = landmarksMaxVSL[0][1];
		float xRightEye = landmarksMaxVSL[0][2];
		float yRightEye = landmarksMaxVSL[0][3];
		float xNose = landmarksMaxVSL[0][4];
		float yNose = landmarksMaxVSL[0][5];
		float xLeftMouth = landmarksMaxVSL[0][6];
		float yLeftMouth = landmarksMaxVSL[0][7];
		float xRightMouth = landmarksMaxVSL[0][8];
		float yRightMouth = landmarksMaxVSL[0][9];
		float center_xNIR = (xLeftEye + xRightEye + xNose + xLeftMouth + xRightMouth) / 5;
		float center_yNIR = (yLeftEye + yRightEye + yNose + yLeftMouth + yRightMouth) / 5;

		float distanceEye = sqrt((xLeftEye - xRightEye) * (xLeftEye - xRightEye) + (yLeftEye - yRightEye) * (yLeftEye - yRightEye));
		float distanceMouse = sqrt((xLeftMouth - xRightMouth) * (xLeftMouth - xRightMouth) + (yLeftMouth - yRightMouth) *(yLeftMouth - yRightMouth));
		float distance = (distanceEye + distanceMouse) / 2;
		float x_diff = 0.381198 * distance + 9.29973;

		bboxMaxNIR.resize(1);
		bboxMaxNIR[0].resize(4);
		landmarksMaxNIR.resize(1);
		landmarksMaxNIR[0].resize(10);
		bboxMaxNIR[0][0] = bboxMaxVSL[0][0] + x_diff;
		bboxMaxNIR[0][1] = bboxMaxVSL[0][1];
		bboxMaxNIR[0][2] = bboxMaxVSL[0][2];
		bboxMaxNIR[0][3] = bboxMaxVSL[0][3];
		landmarksMaxNIR[0][0] = landmarksMaxVSL[0][0] + x_diff;
		landmarksMaxNIR[0][1] = landmarksMaxVSL[0][1];
		landmarksMaxNIR[0][2] = landmarksMaxVSL[0][2] + x_diff;
		landmarksMaxNIR[0][3] = landmarksMaxVSL[0][3];
		landmarksMaxNIR[0][4] = landmarksMaxVSL[0][4] + x_diff;
		landmarksMaxNIR[0][5] = landmarksMaxVSL[0][5];
		landmarksMaxNIR[0][6] = landmarksMaxVSL[0][6] + x_diff;
		landmarksMaxNIR[0][7] = landmarksMaxVSL[0][7];
		landmarksMaxNIR[0][8] = landmarksMaxVSL[0][8] + x_diff;
		landmarksMaxNIR[0][9] = landmarksMaxVSL[0][9];
	}

	std::shared_ptr<tensor<unsigned char>> tensorFaceAreaNIR = getFaceArea(srcColorNIR, channels, height, width, bboxMaxNIR, landmarksMaxNIR, order);
	std::shared_ptr<tensor<unsigned char>> tensorFaceAreaVSL = getFaceArea(srcColorVSL, channels, height, width, bboxMaxVSL, landmarksMaxVSL, order);
	
	if (device_ < 0)
	{
		//black and white paper attack, detect from visible light faceArea
		{
			std::shared_ptr<tensor<unsigned char>> tensorHSVVSL;
			tensor_operation_cpu::rgb2hsv_cpu(tensorFaceAreaVSL, tensorHSVVSL);//tensorHSVVSL is ordered in NCHW
			saturate_data_ = tensorHSVVSL->mutable_cpu_data() + 168 * 128;

			std::vector<int> saturateValue(256);
			for (int i = 0; i < 256; i++)
			{
				saturateValue[i] = 0;
			}

			for (int row = 0; row < 168; ++row)
			{
				const unsigned char* rowData = saturate_data_ + row * 128;
				for (int col = 0; col < 128; ++col)
				{
					int value = static_cast<unsigned char>(rowData[col]);
					saturateValue[value]++;
				}
			}

			//order by probability(ascend), accumulate last 20, bigger than 0.72 is black and white attack(paper or screen)
			unsigned totalNum = 0;
			float totalProbability = 0;
			unsigned pixNum = 168 * 128;
			std::sort(saturateValue.begin(), saturateValue.end());

			for (int i = 0; i < 20; ++i)
			{
				totalNum += saturateValue[255 - i];
			}
			totalProbability = (float)totalNum / pixNum;

			if (totalProbability > 0.72)
			{
				return false;
			}
		}

		//screen attack, detect from near infrared faceArea
		{
			std::shared_ptr<tensor<unsigned char>> tensorGrayNIR, tensorSobelNIR;
			tensor_operation_cpu::rgb2gray_cpu(tensorFaceAreaNIR, tensorGrayNIR);
			tensor_operation_cpu::sobel_cpu(tensorGrayNIR, tensorSobelNIR, 1, 1);
			face_sobel_data_ = tensorSobelNIR->mutable_cpu_data();

			std::vector<int> highFrequencyValue(256);
			for (int i = 0; i < 256; i++)
			{
				highFrequencyValue[i] = 0;
			}

			for (int row = 0; row < 168; ++row)
			{
				const unsigned char* rowData = face_sobel_data_ + row * 128;
				for (int col = 0; col < 128; ++col)
				{
					int value = static_cast<unsigned char>(rowData[col]);
					highFrequencyValue[value]++;
				}
			}

			unsigned totalNum = 0;
			float totalProbability;
			unsigned pixNum = 168 * 128;
			std::sort(highFrequencyValue.begin(), highFrequencyValue.end());

			for (int i = 0; i < 5; ++i)
			{
				totalNum += highFrequencyValue[255 - i];
			}
			totalProbability = (float)totalNum / pixNum;

			if (totalProbability > 0.985)
			{
				return false;//screen detected
			}
		}
	}
	else
	{
		//black and white paper attack, detect from visible light faceArea
		{
			std::shared_ptr<tensor<unsigned char>> tensorHSVVSL;
			tensor_operation_gpu::rgb2hsv_gpu(tensorFaceAreaVSL, tensorHSVVSL);//tensorHSVVSL is ordered in NCHW
			cudaMemcpy((void*)saturate_data_, tensorHSVVSL->gpu_data(), 168 * 128 * sizeof(unsigned char), cudaMemcpyDefault);

			std::vector<int> saturateValue(256);
			for (int i = 0; i < 256; i++)
			{
				saturateValue[i] = 0;
			}

			for (int row = 0; row < 168; ++row)
			{
				const unsigned char* rowData = saturate_data_ + row * 128;
				for (int col = 0; col < 128; ++col)
				{
					int value = static_cast<unsigned char>(rowData[col]);
					saturateValue[value]++;
				}
			}

			//order by probability(ascend), accumulate last 20, bigger than 0.72 is black and white attack(paper or screen)
			unsigned totalNum = 0;
			float totalProbability = 0;
			unsigned pixNum = 168 * 128;
			std::sort(saturateValue.begin(), saturateValue.end());

			for (int i = 0; i < 20; ++i)
			{
				totalNum += saturateValue[255 - i];
			}
			totalProbability = (float)totalNum / pixNum;

			if (totalProbability > 0.72)
			{
				return false;
			}
		}

		//screen attack, detect from near infrared faceArea
		{
			std::shared_ptr<tensor<unsigned char>> tensorGrayNIR, tensorSobelNIR;
			tensor_operation_gpu::rgb2gray_gpu(tensorFaceAreaNIR, tensorGrayNIR);
			tensor_operation_gpu::sobel_gpu(tensorGrayNIR, tensorSobelNIR, 1, 1);
			cudaMemcpy((void*)face_sobel_data_, tensorSobelNIR->gpu_data(), 168 * 128 * sizeof(unsigned char), cudaMemcpyDefault);

			std::vector<int> highFrequencyValue(256);
			for (int i = 0; i < 256; i++)
			{
				highFrequencyValue[i] = 0;
			}

			for (int row = 0; row < 168; ++row)
			{
				const unsigned char* rowData = face_sobel_data_ + row * 128;
				for (int col = 0; col < 128; ++col)
				{
					int value = static_cast<unsigned char>(rowData[col]);
					highFrequencyValue[value]++;
				}
			}

			unsigned totalNum = 0;
			float totalProbability;
			unsigned pixNum = 168 * 128;
			std::sort(highFrequencyValue.begin(), highFrequencyValue.end());

			for (int i = 0; i < 5; ++i)
			{
				totalNum += highFrequencyValue[255 - i];
			}
			totalProbability = (float)totalNum / pixNum;

			if (totalProbability > 0.985)
			{
				return false;//screen detected
			}
		}
	}


	//color image attack
	{
		//TODO:
	}

	return true;//human
}