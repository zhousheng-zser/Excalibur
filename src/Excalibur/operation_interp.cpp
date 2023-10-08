#include "Excalibur/operation_interp.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/operation_resize.hpp"

namespace glasssix
{
    namespace excalibur
    {
        using GetOriginalCoordinateFunc = std::function<float(float, float, float, float, float, float)>;

        enum ResizeCoordinateTransformationMode {
            HALF_PIXEL = 0,
            ASYMMETRIC = 1,
            PYTORCH_HALF_PIXEL = 2,
            TF_HALF_PIXEL_FOR_NN = 3,
            ALIGN_CORNERS = 4,
            TF_CROP_AND_RESIZE = 5,
            CoordinateTransformationModeCount = 6,
        };

        GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
            ResizeCoordinateTransformationMode coordinate_transform_mode) {
            switch (coordinate_transform_mode) {
            case ASYMMETRIC:
                return [](float x_resized, float x_scale, float, float, float, float) {
                    return x_resized / x_scale;
                };
            case PYTORCH_HALF_PIXEL:
                return [](float x_resized, float x_scale, float length_resized, float, float, float) {
                    return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
                };
            case TF_HALF_PIXEL_FOR_NN:
                return [](float x_resized, float x_scale, float, float, float, float) {
                    return (x_resized + 0.5f) / x_scale;
                };
            case ALIGN_CORNERS:
                return [](float x_resized, float, float length_resized, float length_original, float, float) {
                    return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
                };
            case TF_CROP_AND_RESIZE:
                return [](float x_resized, float, float length_resized, float length_original, float roi_start, float roi_end) {
                    auto orig = length_resized > 1
                        ? roi_start * (length_original - 1) + (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                        : 0.5 * (roi_start + roi_end) * (length_original - 1);
                    return static_cast<float>(orig);
                };
            default:  // "half_pixel"
                return [](float x_resized, float x_scale, float, float, float, float) {
                    return ((x_resized + 0.5f) / x_scale) - 0.5f;
                };
            }
        }

        template <typename T>
        void UpsampleBilinear(int64_t batch_size,
            int64_t num_channels,
            int64_t input_height,
            int64_t input_width,
            int64_t output_height,
            int64_t output_width,
            float height_scale,
            float width_scale,
            const std::vector<float>& roi,
            bool use_extrapolation,
            float extrapolation_value,
            const T* XdataBase,
            T* YdataBase,
            GetOriginalCoordinateFunc get_original_coordinate)
        {
            std::vector<float> y_original;
            y_original.reserve(output_height);

            std::vector<float> x_original;
            x_original.reserve(output_width);

            size_t idx_buffer_size = 2 * sizeof(int64_t) * (output_height + output_width);

            size_t scale_buffer_size = 2 * sizeof(float_t) * (output_height + output_width);

            std::vector<std::uint8_t> inx_scale_data_buffer(idx_buffer_size + scale_buffer_size);

            // Get pointers to appropriate memory locations in the scratch buffer
            int64_t* idx_data = reinterpret_cast<int64_t*>(inx_scale_data_buffer.data());

            // input_width is the stride for the height dimension
            int64_t* input_width_mul_y1 = idx_data;
            int64_t* input_width_mul_y2 = input_width_mul_y1 + output_height;

            // stride for width is 1 (no multiplication needed)
            int64_t* in_x1 = input_width_mul_y1 + 2 * output_height;
            int64_t* in_x2 = in_x1 + output_width;

            float* scale_data = reinterpret_cast<float*>(in_x2 + output_width);

            float* dy1 = scale_data;
            float* dy2 = dy1 + output_height;

            float* dx1 = dy1 + 2 * output_height;
            float* dx2 = dx1 + output_width;

            // Start processing
            auto roi_y_start = roi.size() / 2 - 2;
            auto roi_y_end = roi.size() - 2;
            for (int64_t y = 0; y < output_height; ++y) {
                float in_y = height_scale == 1 ? static_cast<float>(y)
                    : get_original_coordinate(static_cast<float>(y), height_scale,
                        static_cast<float>(output_height),
                        static_cast<float>(input_height),
                        roi[roi_y_start], roi[roi_y_end]);
                y_original.emplace_back(in_y);
                in_y = std::max(0.0f, std::min(in_y, static_cast<float>(input_height - 1)));

                const int64_t in_y1 = std::min(static_cast<int64_t>(in_y), input_height - 1);
                const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
                dy1[y] = std::fabs(in_y - in_y1);
                dy2[y] = std::fabs(in_y - in_y2);

                if (in_y1 == in_y2) {
                    dy1[y] = 0.5f;
                    dy2[y] = 0.5f;
                }

                input_width_mul_y1[y] = input_width * in_y1;
                input_width_mul_y2[y] = input_width * in_y2;
            }

            auto roi_x_start = roi.size() / 2 - 1;
            auto roi_x_end = roi.size() - 1;
            for (int64_t x = 0; x < output_width; ++x) {
                float in_x = width_scale == 1 ? static_cast<float>(x)
                    : get_original_coordinate(static_cast<float>(x),
                        width_scale,
                        static_cast<float>(output_width),
                        static_cast<float>(input_width),
                        roi[roi_x_start], roi[roi_x_end]);
                x_original.emplace_back(in_x);
                in_x = std::max(0.0f, std::min(in_x, static_cast<float>(input_width - 1)));

                in_x1[x] = std::min(static_cast<int64_t>(in_x), input_width - 1);
                in_x2[x] = std::min(in_x1[x] + 1, input_width - 1);

                dx1[x] = std::fabs(in_x - in_x1[x]);
                dx2[x] = std::fabs(in_x - in_x2[x]);
                if (in_x1[x] == in_x2[x]) {
                    dx1[x] = 0.5f;
                    dx2[x] = 0.5f;
                }
            }

            for (int64_t n = 0; n < batch_size; ++n)
            {
                for (size_t c = 0; c < num_channels; c++)
                {
                    const T* Xdata = XdataBase + (n * num_channels + c) * (input_height * input_width);
                    T* Ydata = YdataBase + (n * num_channels + c) * (output_height * output_width);
                    for (int64_t y = 0; y < output_height; ++y) {
                        for (int64_t x = 0; x < output_width; ++x) {
                            // when use_extrapolation is set and original index of x or y is out of the dim range
                            // then use extrapolation_value as the output value.
                            if (use_extrapolation &&
                                ((y_original[y] < 0 || y_original[y] > static_cast<float>(input_height - 1)) ||
                                    (x_original[x] < 0 || x_original[x] > static_cast<float>(input_width - 1)))) {
                                Ydata[output_width * y + x] = static_cast<T>(extrapolation_value);
                                continue;
                            }

                            T X11 = Xdata[input_width_mul_y1[y] + in_x1[x]];
                            T X21 = Xdata[input_width_mul_y1[y] + in_x2[x]];
                            T X12 = Xdata[input_width_mul_y2[y] + in_x1[x]];
                            T X22 = Xdata[input_width_mul_y2[y] + in_x2[x]];

                            Ydata[output_width * y + x] = static_cast<T>(dx2[x] * dy2[y] * X11 +
                                dx1[x] * dy2[y] * X21 +
                                dx2[x] * dy1[y] * X12 +
                                dx1[x] * dy1[y] * X22);
                        }
                    }
                    Xdata += input_height * input_width;
                    Ydata += output_width * output_height;
                }
            }
        }

        template <class Dtype>
        operation_interp<Dtype>::operation_interp(const operation_param &param) : operation<Dtype>(param), output_width_(0), output_height_(0)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->resize_type_ = std::stoi(kvs[1]);
                    break;
                case 1:
                    this->width_scale_ = std::stof(kvs[1]);
                    break;
                case 2:
                    this->height_scale_ = std::stof(kvs[1]);
                    break;
                case 3:
                    this->output_width_ = std::stoi(kvs[1]);
                    break;
                case 4:
                    this->output_height_ = std::stoi(kvs[1]);
                    break;
                case 6:
                    this->align_corner_ = std::stoi(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported Interp Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_interp<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_GE(bottoms.size(), 1);
            CHECK_LE(bottoms.size(), 2);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();
            int bottom_w = bottoms[0]->width();
            int bottom_h = bottoms[0]->height();
            int bottom_c = bottoms[0]->channels();
            int outw = output_width_;
            int outh = output_height_;
            if (output_width_ == 0 || output_height_ == 0)
            {
                if (bottoms.size() == 1)
                {
                    outh = static_cast<int>(bottom_h * height_scale_);
                    outw = static_cast<int>(bottom_w * width_scale_);
                }
                else
                {
                    outh = bottoms[1]->cpu_data()[2];
                    outw = bottoms[1]->cpu_data()[3];
                }
            }
            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, bottom_c, outh, outw}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            // nearest
            if (resize_type_ == 1 && align_corner_ == 0)
            {
                const float hs = outh ? bottom_h / (float)outh : 1.f / height_scale_;
                const float ws = outw ? bottom_w / (float)outw : 1.f / width_scale_;

                for (int n = 0; n < num; n++) {
                    for (int q = 0; q < bottom_c; q++)
                    {
                        const float* ptr = bottoms[0]->cpu_data() + bottoms[0]->offset(n, q);
                        float* outptr = tops[0]->mutable_cpu_data() + tops[0]->offset(n, q);
                        for (int y = 0; y < outh; y++)
                        {
                            int in_y = std::min((int)(y * hs), (bottom_h - 1));
                            for (int x = 0; x < outw; x++)
                            {
                                int in_x = std::min((int)(x * ws), (bottom_w - 1));
                                *outptr++ = ptr[in_y * bottom_w + in_x];
                            }
                        }
                    }
                }
                return;
            }
            else if (resize_type_ == 2 && align_corner_ == 1)
            {
                std::vector<float> roi{0, 0, 0, 0, 1, 1, 1, 1};
                float height_scale = outh * 1.0f / bottom_h;
                float width_scale = outw * 1.0f / bottom_w;
                UpsampleBilinear<float>(num, bottom_c, bottom_h, bottom_w, outh, outw, height_scale, width_scale, roi, false, 0.f, bottoms[0]->cpu_data(), tops[0]->mutable_cpu_data(), GetOriginalCoordinateFromResizedCoordinate(ALIGN_CORNERS));
            }
            else if (resize_type_ == 2 && align_corner_ == 0)
            {
                resize_cpu(bottoms[0], tops[0], outh, outw, interpolationType(resize_type_ - 1));
            }
            else
            {
                // other interpolation methods}
                NOT_IMPLEMENTED;
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_interp)
#endif

        INSTANCE_CLASS(operation_interp);
        REGISTE(operation_interp);
    }
}