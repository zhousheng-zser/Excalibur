#include "ultra_face.hpp"
#include "../../include/Primitives/profiler.hpp"
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace glasssix
{
	ultra_face::ultra_face(const std::string param_path, const std::string bin_path,
		int input_width, int input_length, int num_thread_, float score_threshold_, float iou_threshold_, int topk_)
	{
		num_thread = num_thread_;
		topk = topk_;
		score_threshold = score_threshold_;
		iou_threshold = iou_threshold_;
		in_w = input_width;
		in_h = input_length;
		w_h_list = { in_w, in_h };

		for (auto size : w_h_list) 
		{
			std::vector<float> fm_item;
			for (float stride : strides) 
			{
				fm_item.push_back(ceil(size / stride));
			}
			featuremap_size.push_back(fm_item);
		}

		for (auto size : w_h_list) 
		{
			shrinkage_size.push_back(strides);
		}

		/* generate prior anchors */
		for (int index = 0; index < num_featuremap; index++) 
		{
			float scale_w = in_w / shrinkage_size[0][index];
			float scale_h = in_h / shrinkage_size[1][index];
			for (int j = 0; j < featuremap_size[1][index]; j++) 
			{
				for (int i = 0; i < featuremap_size[0][index]; i++) 
				{
					float x_center = (i + 0.5) / scale_w;
					float y_center = (j + 0.5) / scale_h;

					for (float k : min_boxes[index]) 
					{
						float w = k / in_w;
						float h = k / in_h;
						priors.push_back({ clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1) });
					}
				}
			}
		}
		num_anchors = priors.size();
		/* generate prior anchors finished */
		pipe.reset(new excalibur::pipeline<float>(param_path, bin_path, -1));
	}


	ultra_face::~ultra_face()
	{
	}

	std::vector<FaceInfo> ultra_face::detect(cv::Mat &img)
	{
		std::vector<FaceInfo> res;
		if (img.empty()) 
		{
			std::cout << "image is empty ,please check!" << std::endl;
			return std::vector<FaceInfo>{};
		}

		image_h = img.rows;
		image_w = img.cols;

		cv::Mat in;
		cv::resize(img, in, cv::Size(in_w, in_h));

		std::vector<FaceInfo> bbox_collection;
		std::vector<FaceInfo> valid_input;
		std::shared_ptr<memory::tensor<float>> img_t;
		mat2tensor_cpu(in, img_t);
		pipe->forward_cpu(img_t);
		auto scores = pipe->get_featmap("scores");
		for (size_t i = 1000; i < 1100; i++)
		{
			std::cout << scores->cpu_data()[i] << " ";
		}
		std::cout << std::endl;
		auto boxes = pipe->get_featmap("boxes");

		generateBBox(bbox_collection, scores, boxes, score_threshold, num_anchors);
		nms(bbox_collection, res);
		return res;
	}

	void ultra_face::generateBBox(std::vector<FaceInfo> &bbox_collection, 
		std::shared_ptr<memory::tensor<float>> scores, std::shared_ptr<memory::tensor<float>> boxes, float score_threshold, int num_anchors)
	{
		const float* score_data = scores->cpu_data();
		const float* box_data = boxes->cpu_data();
		for (int i = 0; i < num_anchors; i++) 
		{
			if (score_data[i * 2 + 1] > score_threshold)
			{
				FaceInfo rects;
				float x_center = box_data[i * 4] * center_variance * priors[i][2] + priors[i][0];
				float y_center = box_data[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
				float w = exp(box_data[i * 4 + 2] * size_variance) * priors[i][2];
				float h = exp(box_data[i * 4 + 3] * size_variance) * priors[i][3];

				rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
				rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
				rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
				rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
				rects.score = clip(score_data[i * 2 + 1], 1);
				bbox_collection.push_back(rects);
			}
		}
	}

	void ultra_face::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) 
	{
		std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

		int box_num = input.size();

		std::vector<int> merged(box_num, 0);

		for (int i = 0; i < box_num; i++) 
		{
			if (merged[i])
				continue;
			std::vector<FaceInfo> buf;

			buf.push_back(input[i]);
			merged[i] = 1;

			float h0 = input[i].y2 - input[i].y1 + 1;
			float w0 = input[i].x2 - input[i].x1 + 1;

			float area0 = h0 * w0;

			for (int j = i + 1; j < box_num; j++) 
			{
				if (merged[j])
					continue;

				float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
				float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

				float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
				float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

				float inner_h = inner_y1 - inner_y0 + 1;
				float inner_w = inner_x1 - inner_x0 + 1;

				if (inner_h <= 0 || inner_w <= 0)
					continue;

				float inner_area = inner_h * inner_w;

				float h1 = input[j].y2 - input[j].y1 + 1;
				float w1 = input[j].x2 - input[j].x1 + 1;

				float area1 = h1 * w1;

				float score;

				score = inner_area / (area0 + area1 - inner_area);

				if (score > iou_threshold) 
				{
					merged[j] = 1;
					buf.push_back(input[j]);
				}
			}
			switch (type) 
			{
				case hard_nms: 
				{
					output.push_back(buf[0]);
					break;
				}
				case blending_nms: 
				{
					float total = 0;
					for (int i = 0; i < buf.size(); i++) 
					{
						total += exp(buf[i].score);
					}
					FaceInfo rects;
					memset(&rects, 0, sizeof(rects));
					for (int i = 0; i < buf.size(); i++) 
					{
						float rate = exp(buf[i].score) / total;
						rects.x1 += buf[i].x1 * rate;
						rects.y1 += buf[i].y1 * rate;
						rects.x2 += buf[i].x2 * rate;
						rects.y2 += buf[i].y2 * rate;
						rects.score += buf[i].score * rate;
					}
					output.push_back(rects);
					break;
				}
				default: 
				{
					printf("wrong type of nms.");
					exit(-1);
				}
			}
		}
	}

	
}

