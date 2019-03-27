#include <algorithm>
#include <vector>

#include "LonginusDetector.hpp"
#include "ImageOperation.hpp"
#include "InternalLonginusCascade.hpp"
#include "../../include/Romancia/banshee.hpp"
#include "../../include/Damocles/mtcnn.hpp"

using namespace glasssix::longinus;
using namespace glasssix::excalibur;

LonginusDetector::LonginusDetector(int device): device_(device)
{
	cascades_ = new std::vector<std::shared_ptr<BaseLonginusCascade>>();
	bansheelia_.reset(new Banshee(device_));
	diodorus_.reset(new MTCNN(device_));
	matcher_.reset(new Matcher());
	int dstep = (((48 * 8 + 7) / 8) * 4 - 1) &(~(4 - 1));
	data_.resize(dstep * 48);
}
LonginusDetector::~LonginusDetector()
{
	delete cascades_;
}

std::vector<FaceRect> LonginusDetector::detect(unsigned char *gray, int width, int height, int step, int minSize, float scale, int min_neighbors,
	bool useMultiThreads, bool doEarlyReject)
{
	//LOG_IF(WARNING, (doEarlyReject && device_ >= 0)) << "doEarlyReject dose not work on gpu. It will not take any benifit on speed. Automatic disable.";
	LOG_IF(WARNING, (useMultiThreads && device_ >= 0)) << "useMultiThreads is invalide when working on gpu. Automatic ignore.";

	tensor<unsigned char> tensor_gray({ 1, 1, height, width }, device_);

	if (width * sizeof(unsigned char) == step)
		memcpy(tensor_gray.mutable_cpu_data(), gray, tensor_gray.count() * sizeof(unsigned char));
	else
	{
		unsigned char * dst = tensor_gray.mutable_cpu_data();
		for (size_t i = 0; i < height; i++)
			memcpy(dst + i * width, gray + i * step, width * sizeof(unsigned char));
	}

	minSize = std::max((*cascades_)[0]->getWinWidth(), minSize);
	int maxSize = std::min(tensor_gray.width(), tensor_gray.height());

	if (maxSize < minSize)
	{
		// create a structure vector for the output data
		return std::vector<FaceRect>();
	}
	// containers for the detected faces
	std::vector<CandidateRect> candidateRects;

	int scale_factor1024x = scale * 1024;
	int factor1024x = (minSize << 10) / (*cascades_)[0]->getWinWidth();
	int factor1024x_max = (maxSize << 10) / (*cascades_)[0]->getWinWidth(); //do not round it, to avoid the scan window be out of range

	std::vector<ScaledMatrix> integral_pyramids;

	for (; factor1024x <= factor1024x_max; factor1024x = ((factor1024x*scale_factor1024x + 512) >> 10))
	{
		int dwidth = ((tensor_gray.width() << 10) + factor1024x / 2) / factor1024x;
		int dheight = ((tensor_gray.height() << 10) + factor1024x / 2) / factor1024x;
		int dstep = (((dwidth * 8 + 7) / 8) + 4 - 1) & (~(4 - 1));

		tensor<unsigned char> tensor_psmall;
		if (device_ < 0)
		{
			//tensor_psmall = tensor<unsigned char>({ 1, 1, dheight, dwidth }, device_);
			//resize_cpu_bilinear(gray.cpu_data(), gray.height(), gray.width(), gray.channels(), tensor_psmall.mutable_cpu_data(), dheight, dwidth);

			tensor_psmall = tensor<unsigned char>({ 1, 1, dheight, dstep }, device_);
			myResize(tensor_gray.cpu_data(), tensor_gray.width(), tensor_gray.height(), tensor_gray.width(),
				tensor_psmall.mutable_cpu_data(), dwidth, dheight, dstep);
		}
		else
		{
			tensor_psmall = tensor<unsigned char>({ 1, 1, dheight, dwidth }, device_);
#ifdef USE_CUDA
			resize_gpu_bilinear(tensor_gray.gpu_data(), tensor_gray.height(), tensor_gray.width(), tensor_gray.channels(), tensor_psmall.mutable_gpu_data(), dheight, dwidth, device_);
#else
			NO_GPU;
#endif
		}

		int sum_width = dwidth + 1;
		int sum_height = dheight + 1;

		tensor<int> I({ 1, 1, sum_height, sum_width }, device_);
		if (device_ < 0)
		{
			//myIntegral(tensor_psmall.cpu_data(), dwidth, dheight, dwidth, I.mutable_cpu_data(), I.width());
			myIntegral(tensor_psmall.cpu_data(), dwidth, dheight, dstep, I.mutable_cpu_data(), I.width());
		}
		else
		{
#ifdef USE_CUDA
			integral_gpu(tensor_psmall.gpu_data(), tensor_psmall.width(), tensor_psmall.height(), I.mutable_gpu_data(), sum_width, sum_height);
#else
			NO_GPU;
#endif
		}

		int winStep = ((factor1024x <= 3072) + 1) * 4;
		integral_pyramids.emplace_back(factor1024x, winStep);

		for (size_t i = 0; i < cascades_->size(); i++)
		{
			std::vector<CandidateRect> tempCandidateRects;
			std::dynamic_pointer_cast<InternalLonginusCascade>((*cascades_)[i])->SingleScaleDetect(I, winStep, factor1024x, tempCandidateRects, useMultiThreads, doEarlyReject);
			for (size_t j = 0; j < tempCandidateRects.size(); j++)
			{
				tempCandidateRects[j].index_in_image_pyramids = integral_pyramids.size() - 1;
				tempCandidateRects[j].cascade = (*cascades_)[i];
			}
			candidateRects.insert(candidateRects.end(), tempCandidateRects.begin(), tempCandidateRects.end());
		}
	}

	std::vector<FaceRect> rects;
	for (size_t i = 0; i < candidateRects.size(); i++)
	{
		rects.push_back(candidateRects[i]);
	}
	GroupRects(rects, min_neighbors);
	return rects;
}

std::vector<FaceRectwithFaceInfo> LonginusDetector::detect(unsigned char *gray, int width, int height, int step, int minSize, float scale, int min_neighbors,
	int order, bool useMultiThreads, bool doEarlyReject)
{
	std::vector<FaceRect> rects = detect(gray, width, height, step, minSize, scale, min_neighbors, useMultiThreads, doEarlyReject);
		
	std::vector<std::vector<float> > infoParam;
	std::shared_ptr<excalibur::tensor<unsigned char>> rect_tensor, rect48_tensor, group_rect_tensor;
	group_rect_tensor.reset(new tensor<unsigned char>(std::vector<int>{(int)rects.size(), 1, 48, 48}, device_));

	if (device_ < 0)
	{
		for (size_t i = 0; i < rects.size(); i++)
		{
			rect_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, 1, rects[i].height, rects[i].width}, device_));
			unsigned char* rect_tensor_data = rect_tensor->mutable_cpu_data();

			for (size_t row = 0; row < rects[i].height; row++)
			{
				memcpy(rect_tensor_data + row * rects[i].width, gray + (rects[i].y + row) * step + rects[i].x, rects[i].width * sizeof(unsigned char));
			}
			tensor_operation_cpu::resize_cpu(rect_tensor, rect48_tensor, 48, 48);
			memcpy(group_rect_tensor->mutable_cpu_data() + i * 1 * 48 * 48 * sizeof(unsigned char), rect48_tensor->cpu_data(), 1 * 48 * 48 * sizeof(unsigned char));
		}

		bansheelia_->Forward(group_rect_tensor->cpu_data(), rects.size(), order);
		bansheelia_->getParam(infoParam, rects.size());
	}
	else
	{
#ifdef USE_CUDA
		for (size_t i = 0; i < rects.size(); i++)
		{
			rect_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, 1, rects[i].height, rects[i].width}, device_));
			unsigned char* rect_tensor_data = rect_tensor->mutable_gpu_data();

			for (size_t row = 0; row < rects[i].height; row++)
			{
				cudaMemcpy(rect_tensor_data + row * rects[i].width, gray + (rects[i].y + row) * step + rects[i].x, rects[i].width * sizeof(unsigned char), cudaMemcpyHostToDevice);
			}
			tensor_operation_gpu::resize_gpu(rect_tensor, rect48_tensor, 48, 48);
			cudaMemcpy(group_rect_tensor->mutable_gpu_data() + i * 1 * 48 * 48 * sizeof(unsigned char), rect48_tensor->gpu_data(), 1 * 48 * 48 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
		}

		bansheelia_->Forward(group_rect_tensor->gpu_data(), rects.size(), order);
		bansheelia_->getParam(infoParam, rects.size());
#else
		NO_GPU;
#endif
	}

	std::vector<FaceRectwithFaceInfo> rectsWithLandmark;
	rectsWithLandmark.resize(rects.size());
	for (size_t i = 0; i < rects.size(); i++)
	{
		rectsWithLandmark[i].x = rects[i].x;
		rectsWithLandmark[i].y = rects[i].y;
		rectsWithLandmark[i].height = rects[i].height;
		rectsWithLandmark[i].width = rects[i].width;

		rectsWithLandmark[i].confidence = infoParam[i][0];

		rectsWithLandmark[i].yaw = infoParam[i][1] * 90;
		rectsWithLandmark[i].pitch = infoParam[i][2] * 90;
		rectsWithLandmark[i].roll = infoParam[i][3] * 90;

		for (size_t j = 0; j < 5; j++)
		{
			rectsWithLandmark[i].pts[j].x = infoParam[i][4 + 2 * j] * rectsWithLandmark[i].width + rectsWithLandmark[i].x;
			rectsWithLandmark[i].pts[j].y = infoParam[i][4 + 2 * j + 1] * rectsWithLandmark[i].height + rectsWithLandmark[i].y;
		}
	}

	return rectsWithLandmark;
}

#ifndef RELEASE_SDK
void LonginusDetector::load(std::vector<std::string> cascades, int device)
{
	if (device >= 0)
	{
#ifndef USE_CUDA
		NO_GPU;
#endif
	}

	int win_width = 0;
	int win_height = 0;

	cascades_->resize(cascades.size());
	for (size_t i = 0; i < cascades.size(); i++)
	{
		(*cascades_)[i].reset(new InternalLonginusCascade());
		std::dynamic_pointer_cast<InternalLonginusCascade>((*cascades_)[i])->LoadCascade(cascades[i], device);

		win_width = (*cascades_)[i]->getWinWidth();
		win_height = (*cascades_)[i]->getWinHeight();

		LOG_IF(ERROR, (*cascades_)[i]->isEmpty()) << "cascade " << i << " is empty!";
		LOG_IF(ERROR, ((win_width <= 0) || (win_height <= 0))) << "cascade " << i << " win_width <=0 || win_height <=0.";
		LOG_IF(WARNING, win_width != win_height) << "cascade " << i << " win_width != win_height.";
	}

	device_ = device;
}
#endif

void LonginusDetector::set(DetectionType detectionType, int device)
{
	if (device >= 0)
	{
#ifndef USE_CUDA
		NO_GPU;
#endif
	}

	device_ = device;
	cascades_->resize(0);
	bansheelia_.reset(new Banshee(device_));
	diodorus_.reset(new MTCNN(device_));
	InternalLonginusCascade * cascade = nullptr;
	switch (detectionType)
	{
	case MULTIVIEW:
		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::FRONTAL_REINFORCE, device);
		cascades_->emplace_back(cascade);

		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::LEFT_PROFILE, device);
		cascades_->emplace_back(cascade);

		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::RIGHT_PROFILE, device);
		cascades_->emplace_back(cascade);
		break;
	case MULTIVIEW_REINFORCE:
		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::FRONTAL_REINFORCE, device);
		cascades_->emplace_back(cascade);

		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::LEFT_PROFILE_REINFORCE, device);
		cascades_->emplace_back(cascade);

		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::RIGHT_PROFILE_REINFORCE, device);
		cascades_->emplace_back(cascade);
		break;
	case FRONTALVIEW:
		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::FRONTAL, device);
		cascades_->emplace_back(cascade);
		break;
	case FRONTALVIEW_REINFORCE:
		cascade = new InternalLonginusCascade();
		cascade->LoadCascade(LonginusCascadeType::FRONTAL_REINFORCE, device);
		cascades_->emplace_back(cascade);
		break;
	default:
		break;
	}
}


std::vector<Match_Retval> LonginusDetector::match(std::vector<FaceRect> &faceRect, const int frame_extract_frequency) const
{
	return matcher_->match(faceRect, frame_extract_frequency);
}

std::vector<Match_Retval> LonginusDetector::match(std::vector<FaceRectwithFaceInfo> &faceRectInfo, const int frame_extract_frequency) const
{
	std::vector<FaceRect> faceRect;
	for (auto i = 0; i < faceRectInfo.size(); i++)
	{
		faceRect.push_back(FaceRect(faceRectInfo[i].x, faceRectInfo[i].y, faceRectInfo[i].width, faceRectInfo[i].height, 
			faceRectInfo[i].neighbors, faceRectInfo[i].confidence));
	}
	return matcher_->match(faceRect, frame_extract_frequency);
}

std::vector<unsigned char> LonginusDetector::alignFace(const unsigned char* ori_image, int n, int channels, int height, int width,
	std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks) const
{
	return bansheelia_->alignFace(ori_image, n, channels, height, width, bbox, landmarks);
}


std::vector<FaceRectwithFaceInfo> LonginusDetector::detectEx(const unsigned char* image, const int channels, const int height, const int width,
	const int minSize, const float* threshold, const float factor, const int stage) const
{
	std::vector<FaceRectwithFaceInfo> output;
	auto res =  diodorus_->Detect(image, channels, height, width, minSize, threshold, factor, stage, 1);
	for (auto i = 0; i < res.size(); i++)
	{
		float w = res[i].bbox.xmax - res[i].bbox.xmin;
		float h = res[i].bbox.ymax - res[i].bbox.ymin;
		FaceRectwithFaceInfo info(FaceRect(res[i].bbox.xmin + w / 2 - h / 2, res[i].bbox.ymin, h, h, 0, res[i].bbox.score));
		info.yaw = 0.0f;
		info.pitch = 0.0f;
		info.roll = 0.0f;
		for (auto j = 0; j < 5; j++)
		{
			info.pts[j] = Point2f(res[i].landmark[2 * j], res[i].landmark[2 * j + 1]);
		}
		output.push_back(info);
	}
	return output;
}
