
#ifdef USE_OPENCV

#include "athene.hpp"

#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <urlmon.h>
#include <queue>
#include <mutex>

#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include <caffe/layers/softmax_layer.hpp>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include <caffe/layers/memory_data_layer.hpp>
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include <caffe/layers/split_layer.hpp>
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/axpy_layer.hpp"
#include "caffe/layers/concat_layer.hpp"

#include "simple_window.hpp"
#include "video_array_renderer.hpp"
#include "sampling_data_info.hpp"
#include "net_camera.hpp"
#include <string>
#include <thread>
#include <Windows.h>
#include "apc.hpp"
#include "../../include/Excalibur/tensor_operation_cpu.hpp"

using namespace cv;
using namespace std;
using namespace glasssix::hippogriff;
using namespace glasssix::excalibur;
using namespace glasssix::ozymandias;

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

HANDLE main_thread = OpenThread(THREAD_ALL_ACCESS, FALSE, GetCurrentThreadId());

struct thread_data
{
	void* any;
	const sampling_data_info& info;
};

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(MemoryDataLayer);
	//REGISTER_LAYER_CLASS(MemoryData);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	//DepthwiseConvolution
	extern INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
	//REGISTER_LAYER_CLASS(DepthwiseConvolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PReLULayer);
	//REGISTER_LAYER_CLASS(PReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	//REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(EltwiseLayer);
	//REGISTER_LAYER_CLASS(Eltwise);
	extern INSTANTIATE_CLASS(SplitLayer);
	//REGISTER_LAYER_CLASS(Split);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	//REGISTER_LAYER_CLASS(BatchNorm);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(SigmoidLayer);
	REGISTER_LAYER_CLASS(Sigmoid);
	extern INSTANTIATE_CLASS(AxpyLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
}

// Use op::round/max/min for basic types (int, char, long, float, double, etc). Never with classes! std:: alternatives uses 'const T&' instead of 'const T' as argument.
// E.g. std::round is really slow (~300 ms vs ~10 ms when I individually apply it to each element of a whole image array (e.g. in floatPtrToUCharCvMat)

template<typename T>
inline int intRound(const T a)
{
	return int(a + 0.5f);
}

// Max/min functions
template<typename T>
inline T fastMax(const T a, const T b)
{
	return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
	return (a < b ? a : b);
}

struct BlobData {
	int count;
	float* list;
	int num;
	int channels;
	int height;
	int width;
	int capacity_count;		//保留空间的元素个数长度，字节数请 * sizeof(float)
};

#define POSE_COCO_COLORS_RENDER_GPU \
	255.f, 0.f, 85.f, \
	255.f, 0.f, 0.f, \
	255.f, 85.f, 0.f, \
	255.f, 170.f, 0.f, \
	255.f, 255.f, 0.f, \
	170.f, 255.f, 0.f, \
	85.f, 255.f, 0.f, \
	0.f, 255.f, 0.f, \
	0.f, 255.f, 85.f, \
	0.f, 255.f, 170.f, \
	0.f, 255.f, 255.f, \
	0.f, 170.f, 255.f, \
	0.f, 85.f, 255.f, \
	0.f, 0.f, 255.f, \
	255.f, 0.f, 170.f, \
	170.f, 0.f, 255.f, \
	255.f, 0.f, 255.f, \
	85.f, 0.f, 255.f

#define POSE_COCO_COLORS_RENDER_GPU2 \
	255.f / 255, 0.f / 255, 85.f / 255, \
	255.f / 255, 0.f / 255, 0.f / 255, \
	255.f / 255, 85.f / 255, 0.f / 255, \
	255.f / 255, 170.f / 255, 0.f / 255, \
	255.f / 255, 255.f / 255, 0.f / 255, \
	170.f / 255, 255.f / 255, 0.f / 255, \
	85.f / 255, 255.f / 255, 0.f / 255, \
	0.f / 255, 255.f / 255, 0.f / 255, \
	0.f / 255, 255.f / 255, 85.f / 255, \
	0.f / 255, 255.f / 255, 170.f / 255, \
	0.f / 255, 255.f / 255, 255.f / 255, \
	0.f / 255, 170.f / 255, 255.f / 255, \
	0.f / 255, 85.f / 255, 255.f / 255, \
	0.f / 255, 0.f / 255, 255.f / 255, \
	255.f / 255, 0.f / 255, 170.f / 255, \
	170.f / 255, 0.f / 255, 255.f / 255, \
	255.f / 255, 0.f / 255, 255.f / 255, \
	85.f / 255, 0.f / 255, 255.f / 255

const std::vector<float> POSE_COCO_COLORS_RENDER{ POSE_COCO_COLORS_RENDER_GPU };
const std::vector<float> POSE_COCO_COLORS_RENDER2{ POSE_COCO_COLORS_RENDER_GPU2 };
const std::vector<unsigned int> POSE_COCO_PAIRS_RENDER{ 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17 };
const unsigned int POSE_MAX_PEOPLE = 96;

//656x368
Mat getImage(const Mat& im, Size baseSize_ = Size(656, 368), float* scale = 0) {
	int w = baseSize_.width;
	int h = baseSize_.height;
	int nh = h;
	float s = h / (float)im.rows;;
	int nw = im.cols * s;

	if (nw > w) {
		nw = w;
		s = w / (float)im.cols;
		nh = im.rows * s;
	}

	if (scale)*scale = 1 / s;
	Rect dst(0, 0, nw, nh);
	Mat bck = Mat::zeros(h, w, CV_8UC3);
	resize(im, bck(dst), Size(nw, nh));
	return bck;
}

//根据得到的结果，连接身体区域
void connectBodyPartsCpu(vector<float>& poseKeypoints, const float* const heatMapPtr, const float* const peaksPtr,
	const Size& heatMapSize, const int maxPeaks, const int interMinAboveThreshold,
	const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor, vector<int>& keypointShape)
{
	keypointShape.resize(3);
	const std::vector<unsigned int> POSE_COCO_PAIRS{ 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17 };
	const std::vector<unsigned int> POSE_COCO_MAP_IDX{ 31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46 };
	const auto& bodyPartPairs = POSE_COCO_PAIRS;
	const auto& mapIdx = POSE_COCO_MAP_IDX;
	const auto numberBodyParts = 18;

	const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

	std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
	const auto subsetCounterIndex = numberBodyParts;
	const auto subsetSize = numberBodyParts + 1;

	const auto peaksOffset = 3 * (maxPeaks + 1);
	const auto heatMapOffset = heatMapSize.area();

	for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
	{
		const auto bodyPartA = bodyPartPairs[2 * pairIndex];
		const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
		const auto* candidateA = peaksPtr + bodyPartA * peaksOffset;
		const auto* candidateB = peaksPtr + bodyPartB * peaksOffset;
		const auto nA = intRound(candidateA[0]);
		const auto nB = intRound(candidateB[0]);

		// add parts into the subset in special case
		if (nA == 0 || nB == 0)
		{
			// Change w.r.t. other
			if (nA == 0) // nB == 0 or not
			{
				for (auto i = 1; i <= nB; i++)
				{
					bool num = false;
					const auto indexB = bodyPartB;
					for (auto j = 0u; j < subset.size(); j++)
					{
						const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
						if (subset[j].first[indexB] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> rowVector(subsetSize, 0);
						rowVector[bodyPartB] = bodyPartB * peaksOffset + i * 3 + 2; //store the index
						rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
						const auto subsetScore = candidateB[i * 3 + 2]; //second last number in each row is the total score
						subset.emplace_back(std::make_pair(rowVector, subsetScore));
					}
				}
			}
			else // if (nA != 0 && nB == 0)
			{
				for (auto i = 1; i <= nA; i++)
				{
					bool num = false;
					const auto indexA = bodyPartA;
					for (auto j = 0u; j < subset.size(); j++)
					{
						const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
						if (subset[j].first[indexA] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> rowVector(subsetSize, 0);
						rowVector[bodyPartA] = bodyPartA * peaksOffset + i * 3 + 2; //store the index
						rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
						const auto subsetScore = candidateA[i * 3 + 2]; //second last number in each row is the total score
						subset.emplace_back(std::make_pair(rowVector, subsetScore));
					}
				}
			}
		}
		else // if (nA != 0 && nB != 0)
		{
			std::vector<std::tuple<double, int, int>> temp;
			const auto numInter = 10;
			const auto* const mapX = heatMapPtr + mapIdx[2 * pairIndex] * heatMapOffset;
			const auto* const mapY = heatMapPtr + mapIdx[2 * pairIndex + 1] * heatMapOffset;
			for (auto i = 1; i <= nA; i++)
			{
				for (auto j = 1; j <= nB; j++)
				{
					const auto dX = candidateB[j * 3] - candidateA[i * 3];
					const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
					const auto normVec = float(std::sqrt(dX*dX + dY * dY));
					// If the peaksPtr are coincident. Don't connect them.
					if (normVec > 1e-6)
					{
						const auto sX = candidateA[i * 3];
						const auto sY = candidateA[i * 3 + 1];
						const auto vecX = dX / normVec;
						const auto vecY = dY / normVec;

						auto sum = 0.;
						auto count = 0;
						for (auto lm = 0; lm < numInter; lm++)
						{
							const auto mX = fastMin(heatMapSize.width - 1, intRound(sX + lm * dX / numInter));
							const auto mY = fastMin(heatMapSize.height - 1, intRound(sY + lm * dY / numInter));
							//checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
							//checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
							const auto idx = mY * heatMapSize.width + mX;
							const auto score = (vecX*mapX[idx] + vecY * mapY[idx]);
							if (score > interThreshold)
							{
								sum += score;
								count++;
							}
						}

						// parts score + connection score
						if (count > interMinAboveThreshold)
							temp.emplace_back(std::make_tuple(sum / count, i, j));
					}
				}
			}

			// select the top minAB connection, assuming that each part occur only once
			// sort rows in descending order based on parts + connection score
			if (!temp.empty())
				std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

			std::vector<std::tuple<int, int, double>> connectionK;

			const auto minAB = fastMin(nA, nB);
			std::vector<int> occurA(nA, 0);
			std::vector<int> occurB(nB, 0);
			auto counter = 0;
			for (auto row = 0u; row < temp.size(); row++)
			{
				const auto score = std::get<0>(temp[row]);
				const auto x = std::get<1>(temp[row]);
				const auto y = std::get<2>(temp[row]);
				if (!occurA[x - 1] && !occurB[y - 1])
				{
					connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
						bodyPartB*peaksOffset + y * 3 + 2,
						score));
					counter++;
					if (counter == minAB)
						break;
					occurA[x - 1] = 1;
					occurB[y - 1] = 1;
				}
			}

			// Cluster all the body part candidates into subset based on the part connection
			// initialize first body part connection 15&16
			if (pairIndex == 0)
			{
				for (const auto connectionKI : connectionK)
				{
					std::vector<int> rowVector(numberBodyParts + 3, 0);
					const auto indexA = std::get<0>(connectionKI);
					const auto indexB = std::get<1>(connectionKI);
					const auto score = std::get<2>(connectionKI);
					rowVector[bodyPartPairs[0]] = indexA;
					rowVector[bodyPartPairs[1]] = indexB;
					rowVector[subsetCounterIndex] = 2;
					// add the score of parts and the connection
					const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
					subset.emplace_back(std::make_pair(rowVector, subsetScore));
				}
			}
			// Add ears connections (in case person is looking to opposite direction to camera)
			else if (pairIndex == 17 || pairIndex == 18)
			{
				for (const auto& connectionKI : connectionK)
				{
					const auto indexA = std::get<0>(connectionKI);
					const auto indexB = std::get<1>(connectionKI);
					for (auto& subsetJ : subset)
					{
						auto& subsetJFirst = subsetJ.first[bodyPartA];
						auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
						if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
							subsetJFirstPlus1 = indexB;
						else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
							subsetJFirst = indexA;
					}
				}
			}
			else
			{
				if (!connectionK.empty())
				{
					// A is already in the subset, find its connection B
					for (auto i = 0u; i < connectionK.size(); i++)
					{
						const auto indexA = std::get<0>(connectionK[i]);
						const auto indexB = std::get<1>(connectionK[i]);
						const auto score = std::get<2>(connectionK[i]);
						auto num = 0;
						for (auto j = 0u; j < subset.size(); j++)
						{
							if (subset[j].first[bodyPartA] == indexA)
							{
								subset[j].first[bodyPartB] = indexB;
								num++;
								subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
								subset[j].second = subset[j].second + peaksPtr[indexB] + score;
							}
						}
						// if can not find partA in the subset, create a new subset
						if (num == 0)
						{
							std::vector<int> rowVector(subsetSize, 0);
							rowVector[bodyPartA] = indexA;
							rowVector[bodyPartB] = indexB;
							rowVector[subsetCounterIndex] = 2;
							const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
							subset.emplace_back(std::make_pair(rowVector, subsetScore));
						}
					}
				}
			}
		}
	}

	// Delete people below the following thresholds:
	// a) minSubsetCnt: removed if less than minSubsetCnt body parts
	// b) minSubsetScore: removed if global score smaller than this
	// c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
	auto numberPeople = 0;
	std::vector<int> validSubsetIndexes;
	validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
	for (auto index = 0u; index < subset.size(); index++)
	{
		const auto subsetCounter = subset[index].first[subsetCounterIndex];
		const auto subsetScore = subset[index].second;
		if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
		{
			numberPeople++;
			validSubsetIndexes.emplace_back(index);
			if (numberPeople == POSE_MAX_PEOPLE)
				break;
		}
		else if (subsetCounter < 1)
			printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
	}

	// Fill and return poseKeypoints
	keypointShape = { numberPeople, (int)numberBodyParts, 3 };
	if (numberPeople > 0)
		poseKeypoints.resize(numberPeople * (int)numberBodyParts * 3);
	else
		poseKeypoints.clear();

	for (auto person = 0u; person < validSubsetIndexes.size(); person++)
	{
		const auto& subsetI = subset[validSubsetIndexes[person]].first;
		for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
		{
			const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
			const auto bodyPartIndex = subsetI[bodyPart];
			if (bodyPartIndex > 0)
			{
				poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
				poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
				poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
			}
			else
			{
				poseKeypoints[baseOffset] = 0.f;
				poseKeypoints[baseOffset + 1] = 0.f;
				poseKeypoints[baseOffset + 2] = 0.f;
			}
		}
	}
}

//bottom_blob是输入，top是输出
void nms(BlobData* bottom_blob, BlobData* top_blob, float threshold) {
	//maxPeaks就是最大人数，+1是为了第一位存个数
	//算法，是每个点，如果大于阈值，同时大于上下左右值的时候，则认为是峰值

	//算法很简单，featuremap的任意一个点，其上下左右和斜上下左右，都小于自身，就认为是要的点
	//然后以该点区域，选择7*7区域，按照得分值和x、y来计算最合适的亚像素坐标

	int w = bottom_blob->width;
	int h = bottom_blob->height;
	int plane_offset = w * h;
	float* ptr = bottom_blob->list;
	float* top_ptr = top_blob->list;
	int top_plane_offset = top_blob->width * top_blob->height;
	int max_peaks = top_blob->height - 1;

	for (int n = 0; n < bottom_blob->num; ++n) {
		for (int c = 0; c < bottom_blob->channels - 1; ++c) {

			int num_peaks = 0;
			for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y) {
				for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x) {
					float value = ptr[y*w + x];
					if (value > threshold) {
						const float topLeft = ptr[(y - 1)*w + x - 1];
						const float top = ptr[(y - 1)*w + x];
						const float topRight = ptr[(y - 1)*w + x + 1];
						const float left = ptr[y*w + x - 1];
						const float right = ptr[y*w + x + 1];
						const float bottomLeft = ptr[(y + 1)*w + x - 1];
						const float bottom = ptr[(y + 1)*w + x];
						const float bottomRight = ptr[(y + 1)*w + x + 1];

						if (value > topLeft && value > top && value > topRight
							&& value > left && value > right
							&& value > bottomLeft && value > bottom && value > bottomRight)
						{
							//计算亚像素坐标
							float xAcc = 0;
							float yAcc = 0;
							float scoreAcc = 0;
							for (int kx = -3; kx <= 3; ++kx) {
								int ux = x + kx;
								if (ux >= 0 && ux < w) {
									for (int ky = -3; ky <= 3; ++ky) {
										int uy = y + ky;
										if (uy >= 0 && uy < h) {
											float score = ptr[uy * w + ux];
											xAcc += ux * score;
											yAcc += uy * score;
											scoreAcc += score;
										}
									}
								}
							}

							xAcc /= scoreAcc;
							yAcc /= scoreAcc;
							scoreAcc = value;
							top_ptr[(num_peaks + 1) * 3 + 0] = xAcc;
							top_ptr[(num_peaks + 1) * 3 + 1] = yAcc;
							top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
							num_peaks++;
						}
					}
				}
			}
			top_ptr[0] = num_peaks;
			ptr += plane_offset;
			top_ptr += top_plane_offset;
		}
	}
}

void renderKeypointsCpu(Mat& frame, const vector<float>& keypoints, vector<int> keyshape, const std::vector<unsigned int>& pairs,
	const std::vector<float> colors, const float thicknessCircleRatio, const float thicknessLineRatioWRTCircle,
	const float threshold, float scale)
{
	// Get frame channels
	const auto width = frame.cols;
	const auto height = frame.rows;
	const auto area = width * height;

	// Parameters
	const auto lineType = 8;
	const auto shift = 0;
	const auto numberColors = colors.size();
	const auto thresholdRectangle = 0.1f;
	const auto numberKeypoints = keyshape[1];

	// Keypoints
	for (auto person = 0; person < keyshape[0]; person++)
	{
		{
			const auto ratioAreas = 1;
			// Size-dependent variables
			const auto thicknessRatio = fastMax(intRound(std::sqrt(area)*thicknessCircleRatio * ratioAreas), 1);
			// Negative thickness in cv::circle means that a filled circle is to be drawn.
			const auto thicknessCircle = (ratioAreas > 0.05 ? thicknessRatio : -1);
			const auto thicknessLine = 2;// intRound(thicknessRatio * thicknessLineRatioWRTCircle);
			const auto radius = thicknessRatio / 2;

			// Draw lines
			for (auto pair = 0u; pair < pairs.size(); pair += 2)
			{
				const auto index1 = (person * numberKeypoints + pairs[pair]) * keyshape[2];
				const auto index2 = (person * numberKeypoints + pairs[pair + 1]) * keyshape[2];
				if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
				{
					const auto colorIndex = pairs[pair + 1] * 3; // Before: colorIndex = pair/2*3;
					const cv::Scalar color{ colors[(colorIndex + 2) % numberColors],
						colors[(colorIndex + 1) % numberColors],
						colors[(colorIndex + 0) % numberColors] };
					const cv::Point keypoint1{ intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale) };
					const cv::Point keypoint2{ intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale) };
					cv::line(frame, keypoint1, keypoint2, color, thicknessLine, lineType, shift);
				}
			}

			// Draw circles
			for (auto part = 0; part < numberKeypoints; part++)
			{
				const auto faceIndex = (person * numberKeypoints + part) * keyshape[2];
				if (keypoints[faceIndex + 2] > threshold)
				{
					const auto colorIndex = part * 3;
					const cv::Scalar color{ colors[(colorIndex + 2) % numberColors],
						colors[(colorIndex + 1) % numberColors],
						colors[(colorIndex + 0) % numberColors] };
					const cv::Point center{ intRound(keypoints[faceIndex] * scale), intRound(keypoints[faceIndex + 1] * scale) };
					cv::circle(frame, center, radius, color, thicknessCircle, lineType, shift);
				}
			}
		}
	}
}

void renderPoseKeypointsCpu(Mat& frame, const vector<float>& poseKeypoints, vector<int> keyshape,
	const float renderThreshold, float scale, const bool blendOriginalFrame = true)
{
	// Background
	if (!blendOriginalFrame)
		frame.setTo(0.f); // [0-255]

						  // Parameters
	const auto thicknessCircleRatio = 1.f / 75.f;
	const auto thicknessLineRatioWRTCircle = 0.75f;
	const auto& pairs = POSE_COCO_PAIRS_RENDER;

	// Render keypoints
	renderKeypointsCpu(frame, poseKeypoints, keyshape, pairs, POSE_COCO_COLORS_RENDER, thicknessCircleRatio,
		thicknessLineRatioWRTCircle, renderThreshold, scale);
}

void getPoseKeypoints(std::vector<int> &lines, std::vector<int> &circles, const vector<float>& keypoints, vector<int> keyshape,
	const float threshold, float scale)
{
	lines.clear();
	circles.clear();

	// Parameters
	const auto& pairs = POSE_COCO_PAIRS_RENDER;
	const auto numberKeypoints = keyshape[1];

	// Keypoints
	for (auto person = 0; person < keyshape[0]; person++)
	{
		{
			// Draw lines
			for (auto pair = 0u; pair < pairs.size(); pair += 2)
			{
				const auto index1 = (person * numberKeypoints + pairs[pair]) * keyshape[2];
				const auto index2 = (person * numberKeypoints + pairs[pair + 1]) * keyshape[2];
				if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
				{
					lines.push_back(intRound(keypoints[index1] * scale));
					lines.push_back(intRound(keypoints[index1 + 1] * scale));
					lines.push_back(intRound(keypoints[index2] * scale));
					lines.push_back(intRound(keypoints[index2 + 1] * scale));
				}
			}

			// Draw circles
			for (auto part = 0; part < numberKeypoints; part++)
			{
				const auto faceIndex = (person * numberKeypoints + part) * keyshape[2];
				if (keypoints[faceIndex + 2] > threshold)
				{
					circles.push_back(intRound(keypoints[faceIndex] * scale));
					circles.push_back(intRound(keypoints[faceIndex + 1] * scale));
				}
			}
		}
	}
}

void setGPU(int gpu_id) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	if (gpu_id < 0) {
		Caffe::set_mode(Caffe::CPU);
	}
	else {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpu_id);
	}
#endif
}

BlobData* createBlob_local(int num, int channels, int height, int width) {
	BlobData* blob = new BlobData();
	blob->num = num;
	blob->width = width;
	blob->channels = channels;
	blob->height = height;
	blob->count = num * width*channels*height;
	blob->list = new float[blob->count];
	blob->capacity_count = blob->count;
	return blob;
}

void releaseBlob_local(BlobData** blob) {
	if (blob) {
		BlobData* ptr = *blob;
		if (ptr) {
			if (ptr->list)
				delete[] ptr->list;

			delete ptr;
		}
		*blob = 0;
	}
}

static Net<float>* net_;
static Blob<float>* input_layer_;
std::mutex mutex_;
std::queue<std::function<void()>> forward_handlers_;
static constexpr int forward_message_ = 106666;

namespace glasssix
{
	namespace athene
	{
		Athene::Athene(const char *stream, const char *deploy, const char *caffemodel, int base_height, int base_width, int device)
			:stream_(stream), deploy_(deploy), caffemodel_(caffemodel), base_height_(base_height), base_width_(base_width), device_(device)
		{
			//disable gflags output
			google::InitGoogleLogging("aa");

			// create renderer.
			window_ = new simple_window{ 1280, 720 };
			renderer_ = new glasssix::ozymandias::video_array_renderer{ window_->handle() };
			renderer_->set_array(1, 1);
			//renderer_->switch_to_single_view(0, 0);

			//small size to speed up			
			setGPU(device_);

			net_ = new Net<float>(deploy, TEST);
			net_->CopyTrainedLayersFrom(caffemodel);
			input_layer_ = net_->input_blobs()[0];
			input_size_ = Size(input_layer_->width(), input_layer_->height());
			baseSize_ = Size(base_width_, base_height_);

			nms_out_ = createBlob_local(1, 56, POSE_MAX_PEOPLE + 1, 3);
			input_ = createBlob_local(1, 57, base_height, base_width);
		}

		void Athene::Forward_cv()
		{
			glasssix::hippogriff::net_camera camera{ "Pose Detect", stream_, 25, 25 };
			camera.sampling_handler([](void* any, const sampling_data_info& info)
			{
				double time_begin1 = getTickCount();

				auto profiler = (Athene*)any;
				int frame_height = info.height();
				int frame_width = info.width();
				auto frame_data = info.data();
				unsigned char *input_data = new unsigned char[frame_height * frame_width * 3];

				for (int row = 0; row < frame_height; row++)
				{
					for (int col = 0; col < frame_width; col++)
					{
						for (int ch = 0; ch < 3; ch++)
						{
							input_data[row * frame_width * 3 + col * 3 + (2 - ch)] = frame_data[row * frame_width * 4 + col * 4 + ch];
						}
					}
				}

				double fee_time1 = (getTickCount() - time_begin1) / getTickFrequency() * 1000;
				printf("total preprocess fee: %.3f ms\n", fee_time1);

				glasssix::apc::queue(main_thread, [=]
				{
					double fee_time3 = (getTickCount() - time_begin1) / getTickFrequency() * 1000;
					printf("interval fee: %.3f, %.3f ms\n", time_begin1 / getTickFrequency() * 1000, fee_time3);

					double time_begin2 = getTickCount();

					cv::Mat input_frame(frame_height, frame_width, CV_8UC3, input_data);
					profiler->Forward(input_frame);
					imshow("Pose Detect", input_frame);
					if ((char(waitKey(1)) == 'q') || (char(waitKey(1)) == 'Q') || (waitKey(1) == 27))
					{
						exit(-1);
					}
					delete input_data;

					double fee_time2 = (getTickCount() - time_begin2) / getTickFrequency() * 1000;
					printf("total post process fee: %.3f ms\n", fee_time2);
				});

			}, this);
			camera.connect();

			while (true)
			{
				SleepEx(INFINITE, TRUE);
			}
		}

		void Athene::Forward(cv::Mat &image)
		{
			float scale = 0;
			vector<float> keypoints;
			vector<int> shape;
			vector<Mat> input_channels;
			if (image.empty())
			{
				return;
			}
			Mat im = getImage(image, baseSize_, &scale);

			//printf("reshape size: %d, %d, %d\n", input_layer->channels(), im.rows, im.cols);
			input_layer_->Reshape(1, input_layer_->channels(), im.rows, im.cols);
			net_->Reshape();
			input_size_ = Size(im.cols, im.rows);

			float* input_data = input_layer_->mutable_cpu_data();
			for (int i = 0; i < input_layer_->channels() * input_layer_->num(); ++i) {
				cv::Mat channel(input_size_.height, input_size_.width, CV_32FC1, input_data);
				input_channels.emplace_back(channel);
				CHECK_EQ((void*)input_data, (void*)channel.data);
				input_data += input_size_.area();
			}

			//获取一帧图片，根据约定的大小，这种方法是为了保证图像的宽高比不变
			im = getImage(image, baseSize_, &scale);

			//这一步转换加减去均值，手动操作
			im.convertTo(im, CV_32F, 1 / 256.f, -0.5);

			split(im, input_channels);

			//double time_begin = getTickCount();
			net_->Forward();

			Blob<float>* net_output_blob = net_->blob_by_name("net_output").get();
			const float* net_output_data_begin = net_output_blob->cpu_data();
			//double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
			//printf("net forward fee: %.3f ms\n", fee_time);

			BlobData* net_output = createBlob_local(net_output_blob->num(), net_output_blob->channels(), net_output_blob->height(), net_output_blob->width());

			//获取网络输出，inplace
			memcpy(net_output->list, net_output_data_begin, net_output_blob->count() * sizeof(float));

			//把heatmap给resize到约定大小
			for (int i = 0; i < net_output->channels; ++i) {
				Mat um(baseSize_.height, baseSize_.width, CV_32F, input_->list + baseSize_.height*baseSize_.width*i);

				//featuremap的resize插值方法很有关系
				resize(Mat(net_output->height, net_output->width, CV_32F, net_output->list + net_output->width*net_output->height*i), um, baseSize_, 0, 0, CV_INTER_CUBIC);
			}

			//获取每个feature map的局部极大值
			nms(input_, nms_out_, 0.05);

			//得到局部极大值后，根据PAFs、points做部件连接
			connectBodyPartsCpu(keypoints, input_->list, nms_out_->list, baseSize_, POSE_MAX_PEOPLE, 9, 0.05, 3, 0.4, 1, shape);

			//printf("render to image.\n");
			//绘图，显示
			renderPoseKeypointsCpu(image, keypoints, shape, 0.05, scale);
			releaseBlob_local(&net_output);
		}

		void Athene::Forward()
		{
			glasssix::hippogriff::net_camera camera{ "Pose Detect", stream_, 25, 5 };

			camera.sampling_handler([](void* any, const sampling_data_info& info)
			{
				auto profiler = (Athene*)any;
				int frame_height = info.height();
				int frame_width = info.width();
				auto frame_data = info.data();
				unsigned char *input_data = new unsigned char[frame_height * frame_width * 3];

				for (int row = 0; row < frame_height; row++)
				{
					for (int col = 0; col < frame_width; col++)
					{
						for (int ch = 0; ch < 3; ch++)
						{
							input_data[row * frame_width * 3 + col * 3 + (2 - ch)] = frame_data[row * frame_width * 4 + col * 4 + ch];
						}
					}
				}

				auto forward_handler = [=]
				{
					struct
					{
						std::vector<int> lines;
						std::vector<int> circles;
					} tmp;

					using tmp_ptr_type = decltype(tmp)*;

					profiler->Forward(input_data, frame_height, frame_width, tmp.lines, tmp.circles);

					auto decorator = profiler->renderer_->get_view_decorator(0, 0);
					decorator.begin_init([](void* any, video_view_decorator& context)
					{
						auto info = static_cast<tmp_ptr_type>(any);

						context.clear();

						int numberColors = POSE_COCO_COLORS_RENDER2.size();
						for (int n = 0; n < info->lines.size() / 4; n++)
						{
							//const auto colorIndex = POSE_COCO_PAIRS_RENDER[2 * n + 4] * 3;
							const auto colorIndex = rand() % 18;
							rgba_color line_color(POSE_COCO_COLORS_RENDER2[(colorIndex + 2) % numberColors],
								                  POSE_COCO_COLORS_RENDER2[(colorIndex + 1) % numberColors],
								                  POSE_COCO_COLORS_RENDER2[(colorIndex + 0) % numberColors]);
							context.add_line(glasssix::ozymandias::line{ float(info->lines[4 * n + 0]),float(info->lines[4 * n + 1]), float(info->lines[4 * n + 2]), float(info->lines[4 * n + 3]), 2, line_color });
						}

						for (int n = 0; n < info->circles.size() / 2; n++)
						{
							const auto colorIndex = rand() % 18;
							//const auto colorIndex = POSE_COCO_PAIRS_RENDER[2 * n + 5] * 3;
							rgba_color circle_color(POSE_COCO_COLORS_RENDER2[(colorIndex + 2) % numberColors],
								                    POSE_COCO_COLORS_RENDER2[(colorIndex + 1) % numberColors],
								                    POSE_COCO_COLORS_RENDER2[(colorIndex + 0) % numberColors]);
							context.add_ellipse(glasssix::ozymandias::ellipse{ float(info->circles[2 * n + 0]),float(info->circles[2 * n + 1]), 10,10,1,circle_color,circle_color });
						}

					}, &tmp);

					delete input_data;
				};

				{
					//std::lock_guard<std::mutex> lock{ profiler->mutex_ };
					//profiler->forward_handlers_.push(forward_handler);

					std::lock_guard<std::mutex> lock{ mutex_ };
					forward_handlers_.push(forward_handler);
				}


				PostThreadMessage(GetThreadId(main_thread), forward_message_, 0, 0);

			}, this);
			camera.connect();

			renderer_->set_data_provider(camera, 0, 0);

			// Message loop
			MSG msg;
			while (GetMessage(&msg, nullptr, 0, 0))
			{
				if (msg.message == forward_message_)
				{
						std::lock_guard<std::mutex> lock{ mutex_ };
						while (!forward_handlers_.empty())
						{
							forward_handlers_.front()();
							forward_handlers_.pop();
						}
				}
				else
				{
					TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
			}
			/*
						while (true)
						{
							SleepEx(INFINITE, TRUE);
						}*/
		}

		void Athene::Forward(unsigned char* image_data, int height, int width, std::vector<int> &lines, std::vector<int> &circles)
		{
			if (image_data == nullptr)
			{
				return;
			}

			std::shared_ptr<tensor<unsigned char>> image_tensor, image_nchw_tensor, image_resize_tensor;
			image_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, 3}, -1, NHWC));
			unsigned char *image_tensor_data = image_tensor->mutable_cpu_data();
			memcpy(image_tensor_data, image_data, height * width * 3 * sizeof(unsigned char));
			tensor_operation_cpu::nhwc2nchw_cpu(image_tensor, image_nchw_tensor);

			float scale = 0;
			vector<float> keypoints;
			vector<int> shape;

			int w = baseSize_.width;
			int h = baseSize_.height;
			int new_h = h;
			float s = h / (float)height;
			int new_w = width * s;

			if (new_w > w) {
				new_w = w;
				s = w / (float)width;
				new_h = height * s;
			}
			scale = 1 / s;

			tensor_operation_cpu::resize_cpu(image_nchw_tensor, image_resize_tensor, new_h, new_w);
			unsigned char *image_resize_tensor_data = image_resize_tensor->mutable_cpu_data();

			//printf("reshape size: %d, %d, %d\n", input_layer->channels(), im.rows, im.cols);
			input_layer_->Reshape(1, input_layer_->channels(), new_h, new_w);
			net_->Reshape();

			float* input_data = input_layer_->mutable_cpu_data();
			for (int i = 0; i < new_h * new_w * 3; i++)
			{
				input_data[i] = float(image_resize_tensor_data[i]) / 256.f - 0.5;
			}

			//double time_begin = getTickCount();
			net_->Forward();

			Blob<float>* net_output_blob = net_->blob_by_name("net_output").get();
			const float* net_output_data_begin = net_output_blob->cpu_data();
			//double fee_time = (getTickCount() - time_begin) / getTickFrequency() * 1000;
			//printf("net forward fee: %.3f ms\n", fee_time);

			BlobData* net_output = createBlob_local(net_output_blob->num(), net_output_blob->channels(), net_output_blob->height(), net_output_blob->width());

			//获取网络输出，inplace
			memcpy(net_output->list, net_output_data_begin, net_output_blob->count() * sizeof(float));

			//把heatmap给resize到约定大小
			for (int i = 0; i < net_output->channels; ++i) {
				Mat um(baseSize_.height, baseSize_.width, CV_32F, input_->list + baseSize_.height*baseSize_.width*i);

				//featuremap的resize插值方法很有关系
				resize(Mat(net_output->height, net_output->width, CV_32F, net_output->list + net_output->width*net_output->height*i), um, baseSize_, 0, 0, CV_INTER_CUBIC);
			}

			//std::shared_ptr<tensor<float>> input_tensor, output_tensor;
			//output_tensor.reset(new tensor<float>(std::vector<int>{1, net_output->channels, net_output->height, net_output->width}));
			//memcpy(output_tensor->mutable_cpu_data(), net_output->list, net_output->channels * net_output->width * net_output->height * sizeof(float));
			//tensor_operation_cpu::resize_cpu(output_tensor, input_tensor, baseSize_.height, baseSize_.width);
			//memcpy(input_->list, input_tensor->cpu_data(), net_output->channels * baseSize_.width * baseSize_.height * sizeof(float));

			//获取每个feature map的局部极大值
			nms(input_, nms_out_, 0.05);

			//得到局部极大值后，根据PAFs、points做部件连接
			connectBodyPartsCpu(keypoints, input_->list, nms_out_->list, baseSize_, POSE_MAX_PEOPLE, 9, 0.05, 3, 0.4, 1, shape);

			//printf("render to image.\n");
			//绘图，显示
			getPoseKeypoints(lines, circles, keypoints, shape, 0.05, scale);

			releaseBlob_local(&net_output);
		}


		Athene::~Athene()
		{
			if (input_)
			{
				releaseBlob_local(&input_);
			}

			if (nms_out_)
			{
				releaseBlob_local(&nms_out_);
			}

			if (renderer_ != nullptr)
			{
				delete renderer_;
			}

			if (window_ != nullptr)
			{
				delete window_;
			}

			delete net_;
		}
	}
}

#endif  // USE_OPENCV