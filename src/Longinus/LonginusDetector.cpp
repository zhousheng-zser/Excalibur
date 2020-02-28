#include <algorithm>
#include <vector>

#ifdef __ANDROID__
#include <task_scheduler.hpp>
#include <business_task_id.hpp>
#endif

#include "LonginusDetector.hpp"
#include "ImageOperation.hpp"
#include "InternalLonginusCascade.hpp"
#include "../../include/Romancia/banshee.hpp"
#include "../../include/Selene/blur_vsl_net.hpp"
#include "../../include/Selene/black_white_vsl.hpp"
#include "../../include/Selene/face_nose_nir.hpp"
#ifndef TRIAL
#include "../../include/Damocles/mtcnn.hpp"
#include "../../include/Damocles/mtcnn_mobile.hpp"
#include "../../include/Damocles/mtcnn_mobile_nir.hpp"
#endif // !TRIAL

namespace glasssix
{
	namespace longinus
	{
		class LonginusDetector::impl
		{
		public:
			impl(int device = -1);
			virtual ~impl();

			std::vector<face_rect_basic> detect(unsigned char* gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, bool useMultiThreads = false, bool doEarlyReject = false);
			std::vector<face_rect_with_face_info> detect(unsigned char* gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, int order = 0, bool useMultiThreads = false, bool doEarlyReject = false);

			std::vector<Match_Retval> match(std::vector<face_rect_basic>& faceRect, const int frame_extract_frequency, float distance_fractor = 1.0f) const;

			std::vector<Match_Retval> match(std::vector<face_rect_with_face_info>& faceRect, const int frame_extract_frequency, float distance_fractor = 1.0f) const;

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width,
				std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks) const;

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width) const;

#ifndef TRIAL
			std::vector<face_rect_with_face_info> detectEx(const unsigned char* image, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, const int order = 1) const;

			std::vector<face_rect_with_face_info> detectEx_mobile(const unsigned char* image, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, const int order = 1) const;

			std::vector<face_rect_with_face_info> detectEx_mobile_nir(const unsigned char* image, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, const int order = 1) const;

			std::vector<std::vector<face_rect_with_face_info>> detectEx_mobile_pair(const unsigned char* vsl_image, const int vsl_channels, const int vsl_height, const int vsl_width,
				const int vsl_minSize, const float* vsl_threshold, const float vsl_factor, const int vsl_stage, const int vsl_order,
				const unsigned char* nir_image, int nir_channels = 0, int nir_height = 0, int nir_width = 0,
				int nir_minSize = 0, const float* nir_threshold = nullptr, float nir_factor = 0, int nir_stage = 0, int nir_order = 1) const;

			bool blur_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const;

			bool black_white_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const;

			bool face_nose_judge_nir(const unsigned char* nir_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const;

#endif // !TRIAL

#ifndef RELEASE_SDK
			void load(std::vector<std::string> cascades, int device = -1);
#endif
			void set(longinus_detection_type detectionType, int device = -1);

			static const char* getVersion();
		private:
			int device_;
			std::vector<std::shared_ptr<BaseLonginusCascade>>* cascades_;
			std::unique_ptr<Banshee> bansheelia_;
			std::vector<unsigned char> data_;
			std::unique_ptr<Matcher> matcher_;

			std::shared_ptr<Selene> selene_blur_vsl_;
			std::shared_ptr<Selene> selene_black_white_vsl_;
			std::shared_ptr<Selene> selene_face_nose_nir_;

#ifndef TRIAL
			std::unique_ptr<vDamocles> diodorus_;
			std::unique_ptr<vDamocles> diodorus_mobile_;
			std::unique_ptr<vDamocles> diodorus_mobile_nir_;
#endif // !TRIAL

		};

		LonginusDetector::impl::impl(int device) : device_{ device }
		{
			cascades_ = new std::vector<std::shared_ptr<BaseLonginusCascade>>();
			bansheelia_.reset(new Banshee(device_));
			selene_blur_vsl_.reset(new Blur_vsl_net(device_));
			selene_black_white_vsl_.reset(new Black_white_vsl(device_));
			selene_face_nose_nir_.reset(new Face_nose_nir(device_));
			matcher_.reset(new Matcher());
			int dstep = (((48 * 8 + 7) / 8) * 4 - 1) & (~(4 - 1));
			data_.resize(dstep * 48);

#ifndef TRIAL
			diodorus_.reset(new MTCNN(device_));
			diodorus_mobile_.reset(new mtcnn_mobile(device_));
			diodorus_mobile_nir_.reset(new mtcnn_mobile_nir(device_));
#endif // !TRIAL
		}
		LonginusDetector::impl::~impl()
		{
			delete cascades_;
		}

		std::vector<face_rect_basic> LonginusDetector::impl::detect(unsigned char* gray, int width, int height, int step, int minSize, float scale, int min_neighbors,
			bool useMultiThreads, bool doEarlyReject)
		{
			//LOG_IF(WARNING, (doEarlyReject && device_ >= 0)) << "doEarlyReject dose not work on gpu. It will not take any benifit on speed. Automatic disable.";
			LOG_IF(WARNING, (useMultiThreads && device_ >= 0)) << "useMultiThreads is invalide when working on gpu. Automatic ignore.";

			tensor<unsigned char> tensor_gray({ 1, 1, height, width }, device_);

			if (width * sizeof(unsigned char) == step)
				memcpy(tensor_gray.mutable_cpu_data(), gray, tensor_gray.count() * sizeof(unsigned char));
			else
			{
				unsigned char* dst = tensor_gray.mutable_cpu_data();
				for (size_t i = 0; i < height; i++)
					memcpy(dst + i * width, gray + i * step, width * sizeof(unsigned char));
			}

			minSize = std::max((*cascades_)[0]->getWinWidth(), minSize);
			int maxSize = std::min(tensor_gray.width(), tensor_gray.height());

			if (maxSize < minSize)
			{
				// create a structure vector for the output data
				return std::vector<face_rect_basic>();
			}
			// containers for the detected faces
			std::vector<CandidateRect> candidateRects;

			int scale_factor1024x = scale * 1024;
			int factor1024x = (minSize << 10) / (*cascades_)[0]->getWinWidth();
			int factor1024x_max = (maxSize << 10) / (*cascades_)[0]->getWinWidth(); //do not round it, to avoid the scan window be out of range

			std::vector<ScaledMatrix> integral_pyramids;

			for (; factor1024x <= factor1024x_max; factor1024x = ((factor1024x * scale_factor1024x + 512) >> 10))
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

			std::vector<face_rect_basic> rects;
			for (size_t i = 0; i < candidateRects.size(); i++)
			{
				rects.push_back(candidateRects[i]);
			}
			GroupRects(rects, min_neighbors);
			return rects;
		}

		std::vector<face_rect_with_face_info> LonginusDetector::impl::detect(unsigned char* gray, int width, int height, int step, int minSize, float scale, int min_neighbors,
			int order, bool useMultiThreads, bool doEarlyReject)
		{
			std::vector<face_rect_basic> rects = detect(gray, width, height, step, minSize, scale, min_neighbors, useMultiThreads, doEarlyReject);
			if (rects.size() == 0)
			{
				return std::vector<face_rect_with_face_info>();
			}
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
						CUDA_CHECK(cudaMemcpy(rect_tensor_data + row * rects[i].width, gray + (rects[i].y + row) * step + rects[i].x, rects[i].width * sizeof(unsigned char), cudaMemcpyHostToDevice));
					}
					tensor_operation_gpu::resize_gpu(rect_tensor, rect48_tensor, 48, 48);
					CUDA_CHECK(cudaMemcpy(group_rect_tensor->mutable_gpu_data() + i * 1 * 48 * 48 * sizeof(unsigned char), rect48_tensor->gpu_data(), 1 * 48 * 48 * sizeof(unsigned char), cudaMemcpyDeviceToDevice));
				}

				bansheelia_->Forward(group_rect_tensor->gpu_data(), rects.size(), order);
				bansheelia_->getParam(infoParam, rects.size());
#else
				NO_GPU;
#endif
			}

			std::vector<face_rect_with_face_info> rectsWithLandmark;
			rectsWithLandmark.clear();
			for (size_t i = 0; i < rects.size(); i++)
			{
				if (infoParam[i][0] < 0.8)
				{
					continue;
				}

				face_rect_with_face_info temp;
				temp.x = rects[i].x;
				temp.y = rects[i].y;
				temp.height = rects[i].height;
				temp.width = rects[i].width;

				temp.confidence = infoParam[i][0];

				temp.yaw = infoParam[i][1] * 90;
				temp.pitch = infoParam[i][2] * 90;
				temp.roll = infoParam[i][3] * 90;

				for (size_t j = 0; j < 5; j++)
				{
					temp.pts[j].x = infoParam[i][4 + 2 * j] * temp.width + temp.x;
					temp.pts[j].y = infoParam[i][4 + 2 * j + 1] * temp.height + temp.y;
				}

				rectsWithLandmark.push_back(temp);
			}
			return rectsWithLandmark;
		}

#ifndef RELEASE_SDK
		void LonginusDetector::impl::load(std::vector<std::string> cascades, int device)
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

		void LonginusDetector::impl::set(longinus_detection_type detectionType, int device)
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
			selene_blur_vsl_.reset(new Blur_vsl_net(device_));
			selene_black_white_vsl_.reset(new Black_white_vsl(device_));
			selene_face_nose_nir_.reset(new Face_nose_nir(device_));

#ifndef TRIAL
			diodorus_.reset(new MTCNN(device_));
			diodorus_mobile_.reset(new mtcnn_mobile(device_));
			diodorus_mobile_nir_.reset(new mtcnn_mobile_nir(device_));
#endif // !TRIAL

			InternalLonginusCascade* cascade = nullptr;
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

		std::vector<Match_Retval> LonginusDetector::impl::match(std::vector<face_rect_basic>& faceRect, const int frame_extract_frequency, float distance_fractor) const
		{
			return matcher_->match(faceRect, frame_extract_frequency, distance_fractor);
		}

		std::vector<Match_Retval> LonginusDetector::impl::match(std::vector<face_rect_with_face_info>& faceRectInfo, const int frame_extract_frequency, float distance_fractor) const
		{
			std::vector<face_rect_basic> faceRect;
			for (auto i = 0; i < faceRectInfo.size(); i++)
			{
				faceRect.push_back(face_rect_basic(faceRectInfo[i].x, faceRectInfo[i].y, faceRectInfo[i].width, faceRectInfo[i].height,
					faceRectInfo[i].neighbors, faceRectInfo[i].confidence));
			}
			return matcher_->match(faceRect, frame_extract_frequency, distance_fractor);
		}

		std::vector<unsigned char> LonginusDetector::impl::alignFace(const unsigned char* ori_image, int n, int channels, int height, int width,
			std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks) const
		{
#ifdef __ANDROID__
			return glasssix::task_scheduler::current().commit(glasssix::business_task_id::extraction_and_alignment, [=]
			{
				return bansheelia_->alignFace(ori_image, n, channels, height, width, bbox, landmarks);
			}).get();
#else
			return bansheelia_->alignFace(ori_image, n, channels, height, width, bbox, landmarks);
#endif
		}

		std::vector<unsigned char> LonginusDetector::impl::alignFace(const unsigned char* ori_image, int n, int channels, int height, int width) const
		{
#ifdef __ANDROID__
			return glasssix::task_scheduler::current().commit(glasssix::business_task_id::extraction_and_alignment, [=]
			{
				return bansheelia_->alignFace(ori_image, n, channels, height, width);
			}).get();
#else
			return bansheelia_->alignFace(ori_image, n, channels, height, width);
#endif
		}

#ifndef TRIAL
		std::vector<face_rect_with_face_info> LonginusDetector::impl::detectEx(const unsigned char* image, const int channels, const int height, const int width,
			const int minSize, const float* threshold, const float factor, const int stage, const int order) const
		{
			std::vector<face_rect_with_face_info> output;

#ifdef __ANDROID__
			auto res = glasssix::task_scheduler::current().commit(glasssix::business_task_id::detection_living_and_blurring, [=]
			{
				return diodorus_->Detect(image, channels, height, width, minSize, threshold, factor, stage, order);
			}).get();
#else
			auto res = diodorus_->Detect(image, channels, height, width, minSize, threshold, factor, stage, order);
#endif

			for (auto i = 0; i < res.size(); i++)
			{

#ifdef ANGLES
				float w = res[i].bbox.xmax - res[i].bbox.xmin;
				float h = res[i].bbox.ymax - res[i].bbox.ymin;
				float x = res[i].bbox.xmin + w / 2 - h / 2;
				float y = res[i].bbox.ymin;
				glasssix::excalibur::rectangle<float> rect = glasssix::excalibur::rectangle<float>(x, y, h, h);
				face_rect_with_face_info info(face_rect_basic(x, y, h, h, 0, res[i].bbox.score));

				//get roll, pitch, yaw, using romancia
				{
					std::shared_ptr<tensor<unsigned char>> src_tensor, src_gray_tensor, gray_resized_tensor;

					if (device_ < 0)
					{
						if (order == NHWC)
						{
							src_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, device_, NHWC));
							memcpy(src_tensor->mutable_cpu_data(), image, channels * height * width * sizeof(unsigned char));
						}
						else if (order == NCHW)
						{
							src_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_, NCHW));
							memcpy(src_tensor->mutable_cpu_data(), image, channels * height * width * sizeof(unsigned char));
						}
						else
						{
							NOT_IMPLEMENTED;
						}

						tensor_operation_cpu::rgb2gray_cpu(src_tensor, src_gray_tensor);
						tensor_operation_cpu::safty_cut_cpu(src_gray_tensor, src_gray_tensor, &rect);
						tensor_operation_cpu::resize_cpu(src_gray_tensor, gray_resized_tensor, 48, 48);
						bansheelia_->Forward(gray_resized_tensor->cpu_data(), 1);
						std::vector<std::vector<float> > keypointParam;
						bansheelia_->getParam(keypointParam, 1);
						res[i].headpose[0] = keypointParam[0][1] * 90;
						res[i].headpose[1] = keypointParam[0][2] * 90;
						res[i].headpose[2] = keypointParam[0][3] * 90;
					}
					else
					{
#ifdef USE_CUDA
						if (order == NHWC)
						{
							src_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, device_, NHWC));
							CUDA_CHECK(cudaMemcpy(src_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault));
						}
						else if (order == NCHW)
						{
							src_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_, NCHW));
							CUDA_CHECK(cudaMemcpy(src_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault));
						}
						else
						{
							NOT_IMPLEMENTED;
						}

						tensor_operation_gpu::rgb2gray_gpu(src_tensor, src_gray_tensor);
						tensor_operation_gpu::safty_cut_gpu(src_gray_tensor, src_gray_tensor, &rect);
						tensor_operation_gpu::resize_gpu(src_gray_tensor, gray_resized_tensor, 48, 48);
						bansheelia_->Forward(gray_resized_tensor->gpu_data(), 1);
						std::vector<std::vector<float> > keypointParam;
						bansheelia_->getParam(keypointParam, 1);
						res[i].headpose[0] = keypointParam[0][1] * 90;
						res[i].headpose[1] = keypointParam[0][2] * 90;
						res[i].headpose[2] = keypointParam[0][3] * 90;
#else
						NO_GPU;
#endif // USE_CUDA
					}
				}

				info.yaw = res[i].headpose[0];
				info.pitch = res[i].headpose[1];
				info.roll = res[i].headpose[2];
#endif // ANGLES		

				float x = res[i].bbox.xmin;
				float y = res[i].bbox.ymin;
				float w = res[i].bbox.xmax - res[i].bbox.xmin;
				float h = res[i].bbox.ymax - res[i].bbox.ymin;

				face_rect_with_face_info info(face_rect_basic(x, y, w, h, 0, res[i].bbox.score));

				for (auto j = 0; j < 5; j++)
				{
					info.pts[j] = Point2f(res[i].landmark[2 * j], res[i].landmark[2 * j + 1]);
				}
				output.push_back(info);
			}
			return output;
		}


		std::vector<face_rect_with_face_info> LonginusDetector::impl::detectEx_mobile(const unsigned char* image, const int channels, const int height, const int width, const int minSize, const float* threshold, const float factor, const int stage, const int order) const
		{
			std::vector<face_rect_with_face_info> output;

#ifdef __ANDROID__
			auto res = glasssix::task_scheduler::current().commit(glasssix::business_task_id::detection_living_and_blurring, [=]
			{
				return diodorus_mobile_->Detect(image, channels, height, width, minSize, threshold, factor, stage, order);
			}).get();
#else
			auto res = diodorus_mobile_->Detect(image, channels, height, width, minSize, threshold, factor, stage, order);
#endif

			for (auto i = 0; i < res.size(); i++)
			{
				float w = res[i].bbox.xmax - res[i].bbox.xmin;
				float h = res[i].bbox.ymax - res[i].bbox.ymin;
				face_rect_with_face_info info(face_rect_basic(res[i].bbox.xmin + w / 2 - h / 2, res[i].bbox.ymin, h, h, 0, res[i].bbox.score));
				info.yaw = res[i].headpose[0];
				info.pitch = res[i].headpose[1];
				info.roll = res[i].headpose[2];

				for (auto j = 0; j < 5; j++)
				{
					info.pts[j] = Point2f(res[i].landmark[2 * j], res[i].landmark[2 * j + 1]);
				}
				output.push_back(info);
			}
			return output;
		}


		std::vector<face_rect_with_face_info> LonginusDetector::impl::detectEx_mobile_nir(const unsigned char* image, const int channels, const int height, const int width, const int minSize, const float* threshold, const float factor, const int stage, const int order) const
		{
			std::vector<face_rect_with_face_info> output;

#ifdef __ANDROID__
			auto res = glasssix::task_scheduler::current().commit(glasssix::business_task_id::nir_detection, [=]
			{
				return diodorus_mobile_nir_->Detect(image, channels, height, width, minSize, threshold, factor, stage, order);
			}).get();
#else
			auto res = diodorus_mobile_nir_->Detect(image, channels, height, width, minSize, threshold, factor, stage, order);
#endif

			for (auto i = 0; i < res.size(); i++)
			{
				float w = res[i].bbox.xmax - res[i].bbox.xmin;
				float h = res[i].bbox.ymax - res[i].bbox.ymin;
				face_rect_with_face_info info(face_rect_basic(res[i].bbox.xmin + w / 2 - h / 2, res[i].bbox.ymin, h, h, 0, res[i].bbox.score));
				info.yaw = res[i].headpose[0];
				info.pitch = res[i].headpose[1];
				info.roll = res[i].headpose[2];
				for (auto j = 0; j < 5; j++)
				{
					info.pts[j] = Point2f(res[i].landmark[2 * j], res[i].landmark[2 * j + 1]);
				}
				output.push_back(info);
			}
			return output;
		}

		std::vector<std::vector<face_rect_with_face_info>> LonginusDetector::impl::detectEx_mobile_pair(const unsigned char* vsl_image, const int vsl_channels, const int vsl_height, const int vsl_width,
			const int vsl_minSize, const float* vsl_threshold, const float vsl_factor, const int vsl_stage, const int vsl_order,
			const unsigned char* nir_image, int nir_channels, int nir_height, int nir_width,
			int nir_minSize, const float* nir_threshold, float nir_factor, int nir_stage, int nir_order) const
		{
			std::vector<std::vector<face_rect_with_face_info>> output;
			std::vector<face_rect_with_face_info> output_vsl = detectEx_mobile(vsl_image, vsl_channels, vsl_height, vsl_width, vsl_minSize, vsl_threshold, vsl_factor, vsl_stage, vsl_order);
			output.push_back(output_vsl);

			if (nir_channels == 0)
			{
				nir_channels = vsl_channels;
				nir_height = vsl_height;
				nir_width = vsl_width;
				nir_minSize = vsl_minSize;
				nir_threshold = vsl_threshold;
				nir_factor = vsl_factor;
				nir_stage = vsl_stage;
				nir_order = vsl_order;
			}

			std::vector<face_rect_with_face_info> output_nir = detectEx_mobile_nir(nir_image, nir_channels, nir_height, nir_width, nir_minSize, nir_threshold, nir_factor, nir_stage, nir_order);
			output.push_back(output_nir);

			return output;
		}
#endif

		bool LonginusDetector::impl::blur_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const
		{
			return selene_blur_vsl_->judge(vsl_color_image, height, width, bbox, landmarks, thresh, value, order);
		}

		bool LonginusDetector::impl::black_white_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const
		{
			return selene_black_white_vsl_->judge(vsl_color_image, height, width, bbox, landmarks, thresh, value, order);
		}

		bool LonginusDetector::impl::face_nose_judge_nir(const unsigned char* nir_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const
		{
			return selene_face_nose_nir_->judge(nir_color_image, height, width, bbox, landmarks, thresh, value, order);
		}

		const char* LonginusDetector::impl::getVersion()
		{
#ifdef TRIAL
			return std::string("Glasssix Trial FaceSDK";
#else
			return "Glasssix";
#endif // TRIAL	
		}

		LonginusDetector::LonginusDetector(int device) : impl_{ new impl{device} }
		{
		}

		LonginusDetector::~LonginusDetector()
		{
			if (impl_ != nullptr)
			{
				delete impl_;
				impl_ = nullptr;
			}
		}

		std::vector<face_rect_basic> LonginusDetector::detect(unsigned char* gray, int width, int height, int step, int minSize, float scale, int minNeighbors, bool useMultiThreads, bool doEarlyReject)
		{
			return impl_->detect(gray, width, height, step, minSize, scale, minNeighbors, useMultiThreads, doEarlyReject);
		}

		std::vector<face_rect_with_face_info> LonginusDetector::detect(unsigned char* gray, int width, int height, int step, int minSize, float scale, int minNeighbors, int order, bool useMultiThreads, bool doEarlyReject)
		{
			return impl_->detect(gray, width, height, step, minSize, scale, minNeighbors, order, useMultiThreads, doEarlyReject);
		}

		std::vector<Match_Retval> LonginusDetector::match(std::vector<face_rect_basic>& faceRect, const int frame_extract_frequency, float distance_fractor) const
		{
			return impl_->match(faceRect, frame_extract_frequency, distance_fractor);
		}

		std::vector<Match_Retval> LonginusDetector::match(std::vector<face_rect_with_face_info>& faceRect, const int frame_extract_frequency, float distance_fractor) const
		{
			return impl_->match(faceRect, frame_extract_frequency, distance_fractor);
		}

		std::vector<unsigned char> LonginusDetector::alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks) const
		{
			return impl_->alignFace(ori_image, n, channels, height, width, bbox, landmarks);
		}

		std::vector<unsigned char> LonginusDetector::alignFace(const unsigned char* ori_image, int n, int channels, int height, int width) const
		{
			return impl_->alignFace(ori_image, n, channels, height, width);
		}

#ifndef TRIAL
		std::vector<face_rect_with_face_info> LonginusDetector::detectEx(const unsigned char* image, const int channels, const int height, const int width, const int minSize, const float* threshold, const float factor, const int stage, const int order) const
		{
			return impl_->detectEx(image, channels, height, width, minSize, threshold, factor, stage, order);
		}

		std::vector<face_rect_with_face_info> LonginusDetector::detectEx_mobile(const unsigned char* image, const int channels, const int height, const int width, const int minSize, const float* threshold, const float factor, const int stage, const int order) const
		{
			return impl_->detectEx_mobile(image, channels, height, width, minSize, threshold, factor, stage, order);
		}

		std::vector<face_rect_with_face_info> LonginusDetector::detectEx_mobile_nir(const unsigned char* image, const int channels, const int height, const int width, const int minSize, const float* threshold, const float factor, const int stage, const int order) const
		{
			return impl_->detectEx_mobile_nir(image, channels, height, width, minSize, threshold, factor, stage, order);
		}

		std::vector<std::vector<face_rect_with_face_info>> LonginusDetector::detectEx_mobile_pair(const unsigned char* vsl_image, const int vsl_channels, const int vsl_height, const int vsl_width, const int vsl_minSize, const float* vsl_threshold, const float vsl_factor, const int vsl_stage, const int vsl_order, const unsigned char* nir_image, int nir_channels, int nir_height, int nir_width, int nir_minSize, const float* nir_threshold, float nir_factor, int nir_stage, int nir_order) const
		{
			return impl_->detectEx_mobile_pair(
				vsl_image, vsl_channels, vsl_height, vsl_width, vsl_minSize, vsl_threshold, vsl_factor, vsl_stage, vsl_order,
				nir_image, nir_channels, nir_height, nir_width, nir_minSize, nir_threshold, nir_factor, nir_stage, nir_order
			);
		}

		bool LonginusDetector::blur_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const
		{
			return impl_->blur_judge_vsl(vsl_color_image, height, width, bbox, landmarks, thresh, value, order);
		}

		bool LonginusDetector::black_white_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const
		{
			return impl_->black_white_judge_vsl(vsl_color_image, height, width, bbox, landmarks, thresh, value, order);
		}

		bool LonginusDetector::face_nose_judge_nir(const unsigned char* nir_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const
		{
			return impl_->face_nose_judge_nir(nir_color_image, height, width, bbox, landmarks, thresh, value, order);
		}
#endif // !TRIAL

#ifndef RELEASE_SDK
		void LonginusDetector::load(std::vector<std::string> cascades, int device)
		{
			impl_->load(cascades, device);
		}
#endif

		void LonginusDetector::set(longinus_detection_type detectionType, int device)
		{
			impl_->set(detectionType, device);
		}

		const char* LonginusDetector::getVersion()
		{
			return impl::getVersion();
		}
	}
}
