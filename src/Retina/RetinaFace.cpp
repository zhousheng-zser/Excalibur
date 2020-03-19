#include <vector>
#include "RetinaFace.hpp"
#include "retina_net.hpp"
#include <glasssix/timer.hpp>
//#define SPLIT_TIME
using namespace glasssix::excalibur;
using namespace glasssix::longinus;

//processing
anchor_win  _whctrs(FaceBox anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.xmax - anchor.xmin + 1;
    win.h = anchor.ymax - anchor.ymin + 1;
    win.x_ctr = anchor.xmin + 0.5 * (win.w - 1);
    win.y_ctr = anchor.ymin + 0.5 * (win.h - 1);

    return win;
}

FaceBox _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    FaceBox anchor;
    anchor.xmin = win.x_ctr - 0.5 * (win.w - 1);
    anchor.ymin = win.y_ctr - 0.5 * (win.h - 1);
    anchor.xmax = win.x_ctr + 0.5 * (win.w - 1);
    anchor.ymax = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

std::vector<FaceBox> _ratio_enum(FaceBox anchor, std::vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
	std::vector<FaceBox> anchors;
    for(size_t i = 0; i < ratios.size(); i++) 
	{
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        FaceBox tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

std::vector<FaceBox> _scale_enum(FaceBox anchor, std::vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
	std::vector<FaceBox> anchors;
    for(size_t i = 0; i < scales.size(); i++) 
	{
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        FaceBox tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

std::vector<FaceBox> generate_anchors(int base_size = 16, std::vector<float> ratios = {0.5, 1, 2},
                                         std::vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    FaceBox base_anchor;
    base_anchor.xmin = 0;
    base_anchor.ymin = 0;
    base_anchor.xmax = base_size - 1;
    base_anchor.ymax = base_size - 1;

	std::vector<FaceBox> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

	std::vector<FaceBox> anchors;
    for(size_t i = 0; i < ratio_anchors.size(); i++) 
	{
	    std::vector<FaceBox> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if(dense_anchor) 
	{
        CHECK_EQ(stride % 2 , 0);
	    std::vector<FaceBox> anchors2 = anchors;
        for(size_t i = 0; i < anchors2.size(); i++) {
            anchors2[i].xmin += stride / 2;
            anchors2[i].ymin += stride / 2;
            anchors2[i].xmax += stride / 2;
            anchors2[i].ymax += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

std::vector<std::vector<FaceBox>> generate_anchors_fpn(bool dense_anchor = false, std::vector<anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

	std::vector<std::vector<FaceBox>> anchors;
    for(size_t i = 0; i < cfg.size(); i++) 
	{
        //stride从小到大[32 16 8]
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
		std::vector<float> ratios = tmp.RATIOS;
		std::vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

		std::vector<FaceBox> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

std::vector<FaceBox> anchors_plane(int height, int width, int stride, std::vector<FaceBox> base_anchors)
{
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

	std::vector<FaceBox> all_anchors;
    for(size_t k = 0; k < base_anchors.size(); k++) {
        for(int ih = 0; ih < height; ih++) {
            int sh = ih * stride;
            for(int iw = 0; iw < width; iw++) {
                int sw = iw * stride;

                FaceBox tmp;
                tmp.xmin = base_anchors[k].xmin + sw;
                tmp.ymin = base_anchors[k].ymin + sh;
                tmp.xmax = base_anchors[k].xmax + sw;
                tmp.ymax = base_anchors[k].ymax + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(std::vector<FaceBox> &boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for(size_t i = 0; i < boxes.size(); i++) 
	{
        if(boxes[i].xmin < 0) 
		{
            boxes[i].xmin = 0;
        }
        if(boxes[i].ymin < 0) 
		{
            boxes[i].ymin = 0;
        }
        if(boxes[i].xmax > width - 1) 
		{
            boxes[i].xmax = width - 1;
        }
        if(boxes[i].ymax > height - 1) 
		{
            boxes[i].ymax = height -1;
        }
//        boxes[i].xmin = std::max<float>(std::min<float>(boxes[i].xmin, width - 1), 0);
//        boxes[i].ymin = std::max<float>(std::min<float>(boxes[i].ymin, height - 1), 0);
//        boxes[i].xmax = std::max<float>(std::min<float>(boxes[i].xmax, width - 1), 0);
//        boxes[i].ymax = std::max<float>(std::min<float>(boxes[i].ymax, height - 1), 0);
    }
}

void clip_boxes(FaceBox &box, int width, int height)
{
    //Clip boxes to image boundaries.
    if(box.xmin < 0) {
        box.xmin = 0;
    }
    if(box.ymin < 0) {
        box.ymin = 0;
    }
    if(box.xmax > width - 1) {
        box.xmax = width - 1;
    }
    if(box.ymax > height - 1) {
        box.ymax = height -1;
    }
//    boxes[i].xmin = std::max<float>(std::min<float>(boxes[i].xmin, width - 1), 0);
//    boxes[i].ymin = std::max<float>(std::min<float>(boxes[i].ymin, height - 1), 0);
//    boxes[i].xmax = std::max<float>(std::min<float>(boxes[i].xmax, width - 1), 0);
//    boxes[i].ymax = std::max<float>(std::min<float>(boxes[i].ymax, height - 1), 0);

}

//######################################################################
//retinaface
//######################################################################

RetinaFace::RetinaFace(int device) : device_(device)
{
	retina_net_.reset(new Retina_net(device_));

	//主干网络选择
	int fmc = 3;

	std::string network = "net3";
	if (network == "ssh" || network == "vgg")
	{
		pixel_means[0] = 103.939;
		pixel_means[1] = 116.779;
		pixel_means[2] = 123.68;
	}
	else if (network == "net3")
	{
		_ratio = { 1.0 };
	}
	else if (network == "net3a")
	{
		_ratio = { 1.0, 1.5 };
	}
	else if (network == "net6")
	{ //like pyramidbox or s3fd
		fmc = 6;
	}
	else if (network == "net5")
	{ //retinaface
		fmc = 5;
	}
	else if (network == "net5a")
	{
		fmc = 5;
		_ratio = { 1.0, 1.5 };
	}
	else if (network == "net4")
	{
		fmc = 4;
	}
	else if (network == "net5a")
	{
		fmc = 4;
		_ratio = { 1.0, 1.5 };
	}
	else
	{
		std::cout << "network setting error" << network << std::endl;
	}

	//anchor配置
	if (fmc == 3)
	{
		_feat_stride_fpn = { 32, 16, 8 };
		anchor_cfg tmp;
		tmp.SCALES = { 32, 16 };
		tmp.BASE_SIZE = 16;
		tmp.RATIOS = _ratio;
		tmp.ALLOWED_BORDER = 9999;
		tmp.STRIDE = 32;
		cfg.push_back(tmp);

		tmp.SCALES = { 8, 4 };
		tmp.BASE_SIZE = 16;
		tmp.RATIOS = _ratio;
		tmp.ALLOWED_BORDER = 9999;
		tmp.STRIDE = 16;
		cfg.push_back(tmp);

		tmp.SCALES = { 2, 1 };
		tmp.BASE_SIZE = 16;
		tmp.RATIOS = _ratio;
		tmp.ALLOWED_BORDER = 9999;
		tmp.STRIDE = 8;
		cfg.push_back(tmp);
	}
	else
	{
		std::cout << "please reconfig anchor_cfg" << network << std::endl;
	}

	bool dense_anchor = false;
	std::vector<std::vector<FaceBox>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
	for (size_t i = 0; i < anchors_fpn.size(); i++)
	{
		std::string key = "stride" + std::to_string(_feat_stride_fpn[i]);
		_anchors_fpn[key] = anchors_fpn[i];
		_num_anchors[key] = anchors_fpn[i].size();
	}
}

RetinaFace::~RetinaFace()
{
	
}

FaceBox RetinaFace::bbox_pred(FaceBox anchor, std::vector<float> regress)
{
	CHECK_EQ(regress.size(), 4);
    FaceBox rect;

    float width = anchor.xmax - anchor.xmin + 1;
    float height = anchor.ymax - anchor.ymin + 1;
    float ctr_x = anchor.xmin + 0.5 * (width - 1.0);
    float ctr_y = anchor.ymin + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.xmin = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.ymin = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.xmax = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.ymax = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

bool RetinaFace::CompareBBox(const FaceInfomation & a, const FaceInfomation & b)
{
    return a.bbox.score > b.bbox.score;
}

std::vector<FaceInfomation> RetinaFace::nms(std::vector<FaceInfomation>& bboxes, float threshold)
{
    std::vector<FaceInfomation> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) 
	{
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        //如果全部执行完则返回
        if (select_idx == num_bbox) 
		{
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        FaceBox select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
        float xmin = static_cast<float>(select_bbox.xmin);
        float ymin = static_cast<float>(select_bbox.ymin);
        float xmax = static_cast<float>(select_bbox.xmax);
        float ymax = static_cast<float>(select_bbox.ymax);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) 
		{
            if (mask_merged[i] == 1)
                continue;

            FaceBox& bbox_i = bboxes[i].bbox;
            float x = std::max<float>(xmin, static_cast<float>(bbox_i.xmin));
            float y = std::max<float>(ymin, static_cast<float>(bbox_i.ymin));
            float w = std::min<float>(xmax, static_cast<float>(bbox_i.xmax)) - x + 1;   //<- float 型不加1
            float h = std::min<float>(ymax, static_cast<float>(bbox_i.ymax)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
            float area_intersect = w * h;

   
            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) 
			{
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}

std::vector<FaceInfomation> RetinaFace::detect(const unsigned char *img_data, int img_channel, int img_height, int img_width, int img_order, float threshold, float scales)
{
#ifdef SPLIT_TIME
	glasssix::Timer calcTime;
	calcTime.Start();
#endif // SPLIT_TIME

	if (img_data == NULL)
	{
		return std::vector<FaceInfomation>();
	}

	std::shared_ptr<tensor<unsigned char>> img_tensor;
	if (img_order == 0)
	{
		img_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, img_channel, img_height, img_width}, device_, NCHW));
	}
	else
	{
		img_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, img_height, img_width, img_channel}, device_, NHWC));
	}

	int ws = (img_width + 31) / 32 * 32;
	int hs = (img_height + 31) / 32 * 32;
	std::shared_ptr<tensor<unsigned char>> img_bordered;
	std::vector<std::vector<std::tuple<std::vector<int>, const float*>>> tuple_result;

	//TODO: uncomment this line, and open MACRO 'SPLIT_TIME', you will find infer_time(gpu version) is 35ms faster on 1920*1080 image, both CUDNN and native-gpu
	//retina_net_.reset(new Retina_net(device_));

	if (device_ < 0)
	{
		memcpy(img_tensor->mutable_cpu_data(), img_data, img_channel * img_height * img_width * sizeof(unsigned char));
		tensor_operation_cpu::make_border_cpu(img_tensor, img_bordered, 0, hs - img_height, 0, ws - img_width);

		if (img_order != 0)
		{
			tensor_operation_cpu::nhwc2nchw_cpu(img_bordered, img_bordered);
		}

#ifdef SPLIT_TIME
		calcTime.Stop();
		double pre_time = calcTime.GetElapsedMilliseconds();
		std::cout << "pre_time:" << pre_time << std::endl;
		calcTime.Start();
#endif // SPLIT_TIME

		tuple_result = retina_net_->Forward(img_bordered);

#ifdef SPLIT_TIME
		calcTime.Stop();
		double infer_time = calcTime.GetElapsedMilliseconds();
		std::cout << "infer_time:" << infer_time << std::endl;
		calcTime.Start();
#endif // SPLIT_TIME

	}
	else
	{

#ifdef USE_CUDA
		cudaMemcpy(img_tensor->mutable_gpu_data(), img_data, img_channel * img_height * img_width * sizeof(unsigned char), cudaMemcpyDefault);
		tensor_operation_gpu::make_border_gpu(img_tensor, img_bordered, 0, hs - img_height, 0, ws - img_width);

		if (img_order != 0)
		{
			tensor_operation_gpu::nhwc2nchw_gpu(img_bordered, img_bordered);
		}

#ifdef SPLIT_TIME
		calcTime.Stop();
		double pre_time = calcTime.GetElapsedMilliseconds();
		std::cout << "pre_time:" << pre_time << std::endl;
		calcTime.Start();
#endif // SPLIT_TIME
		
		tuple_result = retina_net_->Forward(img_bordered);
		
#ifdef SPLIT_TIME
		calcTime.Stop();
		double infer_time = calcTime.GetElapsedMilliseconds();
		std::cout << "infer_time:" << infer_time << std::endl;
		calcTime.Start();
#endif // SPLIT_TIME

#else
		NO_GPU;
#endif // USE_CUDA

	}

	std::vector<FaceInfomation> faceInfo;
	for (size_t i = 0; i < tuple_result.size(); i++)
	{
		std::string key = "stride" + std::to_string(_feat_stride_fpn[i]);
		int stride = _feat_stride_fpn[i];

		std::vector<std::tuple<std::vector<int>, const float*>> temp_tuple = tuple_result[i];

		//score
		std::vector<int> score_shape;
		std::vector<int> data_shape;
		const float *data_pointer;
		std::tie(data_shape, data_pointer) = temp_tuple[0];
		score_shape = data_shape;
		auto score_blob_count = [](std::vector<int> size)->int {int count = 1; for (int i = 0; i < size.size(); i++)count *= size[i]; return count; }(data_shape);
		const float* scoreB = data_pointer + score_blob_count / 2;
		const float* scoreE = scoreB + score_blob_count / 2;
		std::vector<float> score = std::vector<float>(scoreB, scoreE);

		//bbox
		std::tie(data_shape, data_pointer) = temp_tuple[1];
		auto bbox_blob_count = [](std::vector<int> size)->int {int count = 1; for (int i = 0; i < size.size(); i++)count *= size[i]; return count; }(data_shape);
		const float* bboxB = data_pointer;
		const float* bboxE = bboxB + bbox_blob_count;
		std::vector<float> bbox_delta = std::vector<float>(bboxB, bboxE);

		//landmarks
		std::tie(data_shape, data_pointer) = temp_tuple[2];
		auto landmark_blob_count = [](std::vector<int> size)->int {int count = 1; for (int i = 0; i < size.size(); i++)count *= size[i]; return count; }(data_shape);
		const float* landmarkB = data_pointer;
		const float* landmarkE = landmarkB + landmark_blob_count;
		std::vector<float> landmark_delta = std::vector<float>(landmarkB, landmarkE);

		int width = score_shape[3];
		int height = score_shape[2];

		size_t count = width * height;
		size_t num_anchor = _num_anchors[key];

		//存储顺序 h * w * num_anchor
		std::vector<FaceBox> anchors = anchors_plane(height, width, stride, _anchors_fpn[key]);

		for (size_t num = 0; num < num_anchor; num++)
		{
			for (size_t j = 0; j < count; j++)
			{
				//置信度小于阈值跳过
				float conf = score[j + count * num];
				if (conf <= threshold)
				{
					continue;
				}

				float dx = bbox_delta[j + count * (0 + num * 4)];
				float dy = bbox_delta[j + count * (1 + num * 4)];
				float dw = bbox_delta[j + count * (2 + num * 4)];
				float dh = bbox_delta[j + count * (3 + num * 4)];
				std::vector<float> regress = { dx, dy, dw, dh };

				//回归人脸框
				FaceBox rect = bbox_pred(anchors[j + count * num], regress);
				//越界处理
				clip_boxes(rect, ws, hs);

				FaceInfomation tmp;
				rect.score = conf;
				tmp.bbox = rect;

				for (size_t k = 0; k < 5; k++)
				{
					tmp.landmark[2 * k] = landmark_delta[j + count * (num * 10 + k * 2)];
					tmp.landmark[2 * k + 1] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
				}
				//回归人脸关键点

				FaceBox anchor = anchors[j + count * num];
				float box_width = anchor.xmax - anchor.xmin + 1;
				float box_height = anchor.ymax - anchor.ymin + 1;
				float ctr_x = anchor.xmin + 0.5 * (box_width - 1.0);
				float ctr_y = anchor.ymin + 0.5 * (box_height - 1.0);

				for (size_t k = 0; k < 5; k++)
				{
					tmp.landmark[2 * k] = tmp.landmark[2 * k] * box_width + ctr_x;
					tmp.landmark[2 * k + 1] = tmp.landmark[2 * k + 1] * box_height + ctr_y;
				}

				faceInfo.push_back(tmp);
			}
		}
	}

	//排序nms
	faceInfo = nms(faceInfo, nms_threshold);

#ifdef SPLIT_TIME
	calcTime.Stop();
	double post_time = calcTime.GetElapsedMilliseconds();
	std::cout << "post_time:" << post_time << std::endl << std::endl;
#endif // SPLIT_TIME

	return faceInfo;
}
