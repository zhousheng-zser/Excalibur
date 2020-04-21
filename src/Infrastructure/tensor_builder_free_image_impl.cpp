#include "tensor_builder_free_image_impl.hpp"
#include "fi_image_ex.hpp"
#include "init_free_image.hpp"
#include "tensor_conversions.hpp"

namespace glasssix
{
    namespace excalibur
    {
        bitmap_converter_map<tensor_layout> tensor_builder_free_image_impl::float_converters_ =
        {
            { tensor_layout::rgb, [](fi_image_ex& inner) { return inner.convert_to_rgbf(); } },
            { tensor_layout::rgba, [](fi_image_ex& inner) { return inner.convert_to_rgbaf(); } },
            { tensor_layout::grayscale, [](fi_image_ex& inner) { return inner.convert_to_float(); } }
        };

        bitmap_converter_map<tensor_layout> tensor_builder_free_image_impl::uint8_converters_ =
        {
            { tensor_layout::rgb, [](fi_image_ex& inner) { return inner.convert_to_24bits(); } },
            { tensor_layout::rgba, [](fi_image_ex& inner) { return inner.convert_to_32bits(); } },
            { tensor_layout::grayscale, [](fi_image_ex& inner) { return inner.convert_to_grayscale(); } }
        };

        tensor_builder_free_image_impl::tensor_builder_free_image_impl() : device_{ -1 }, order_{ memory::NHWC }, width_{}, height_{}, stride_{}, channels_{}
        {
            init_free_image::instance().invoke();
            image_ = std::make_shared<fi_image_ex>();
        }

        tensor_builder_free_image_impl::~tensor_builder_free_image_impl()
        {
        }

        /// <summary>
        /// Load a bitmap from a file.
        /// </summary>
        /// <param name="path">The path of the file</param>
        /// <returns>True on success</returns>
        bool tensor_builder_free_image_impl::load_from(const std::string& path)
        {
            if (image_->load(path.c_str()))
            {
                update_image_parameters_core();

                return true;
            }

            return false;
        }

        /// <summary>
        /// Load a bitmap from an input stream.
        /// </summary>
        /// <param name="stream">The stream</param>
        /// <returns>True on success</returns>
        bool tensor_builder_free_image_impl::load_from(std::istream& stream)
        {
            // Read the stream.
            std::string buffer{ std::istreambuf_iterator<char>{ stream }, std::istreambuf_iterator<char>{} };
            if (buffer.empty())
            {
                return false;
            }

            // Load the bitmap.
            return load_from(buffer.data(), buffer.size());
        }

        /// <summary>
        /// Load a bitmap from a memory block.
        /// </summary>
        /// <param name="data">The memory block</param>
        /// <param name="size">The size in bytes</param>
        /// <returns>True on success</returns>
        bool tensor_builder_free_image_impl::load_from(const void* data, size_t size)
        {
            // Load the bitmap.
            fipMemoryIO block{ static_cast<uint8_t*>(const_cast<void*>(data)), static_cast<uint32_t>(size) };
            if (image_->loadFromMemory(block))
            {
                update_image_parameters_core();

                return true;
            }

            return false;
        }

        /// <summary>
        /// Save the image to a file.
        /// The encoder is deduced by the file extension automatically.
        /// </summary>
        /// <param name="path">The path of the file</param>
        /// <returns>True success</returns>
        bool tensor_builder_free_image_impl::save_to(const std::string& path)
        {
            if (!image_->isValid())
            {
                return false;
            }

            return image_->save(path.c_str());
        }

        /// <summary>
        /// Set the parameters for building a tensor.
        /// </summary>
        /// <param name="order">The memory order</param>
        void tensor_builder_free_image_impl::tensor_parameters(memory::orderType order)
        {
            order_ = order;
        }

        /// <summary>
        /// Set the parameters for building a tensor.
        /// </summary>
        /// <param name="order">The memory order</param>
        /// <param name="device">The device ID</param>
        void tensor_builder_free_image_impl::tensor_parameters(memory::orderType order, int device)
        {
            order_ = order;
            device_ = device;
        }

        /// <summary>
        /// Create an image from a floating-point tensor.
        /// </summary>
        /// <param name="data">The tensor data</param>
        /// <param name="layout">The tensor layout</param>
        /// <returns>
        /// True: success
        /// False: failure
        ///</returns>
        bool tensor_builder_free_image_impl::from_tensor(const memory::tensor<float>& data, tensor_layout layout)
        {
            FREE_IMAGE_TYPE type;

            switch (layout)
            {
            case tensor_layout::rgb:
                type = FIT_RGBF;
                break;
            case tensor_layout::rgba:
                type = FIT_RGBAF;
                break;
            case tensor_layout::grayscale:
                type = FIT_FLOAT;
                break;
            case tensor_layout::grayscale_3:
                type = FIT_RGBF;
                break;
            default:
                return false;
            }

            // Create a new instance.
            image_ = std::make_shared<fi_image_ex>(type, data.width(), data.height(), static_cast<uint32_t>(sizeof(float) * data.channels() * uint8_bits_));

            if (!image_->isValid())
            {
                return false;
            }

            // Copy the data.
            tensor_helper::copy_to_bitmap(data, image_->accessPixels(), image_->getScanWidth());

            // Floating-float numbers are not supported by the stardard bitmap.
            // Thus, we need to convert it to a standard type.
            return image_->convert_to_standard_type_self();
        }

        /// <summary>
        /// Create a bitmap from a uint8 tensor.
        /// </summary>
        /// <param name="data">The tensor data</param>
        /// <param name="layout">The tensor layout</param>
        /// <returns>
        /// True: success
        /// False: failure
        ///</returns>
        bool tensor_builder_free_image_impl::from_tensor(const memory::tensor<uint8_t>& data, tensor_layout layout)
        {
            switch (layout)
            {
            case tensor_layout::rgb:
            case tensor_layout::rgba:
            case tensor_layout::grayscale:
            case tensor_layout::grayscale_3:
                image_ = std::make_shared<fi_image_ex>(FIT_BITMAP, data.width(), data.height(), static_cast<uint32_t>(sizeof(uint8_t) * data.channels() * uint8_bits_));
                break;
            default:
                return false;
            }

            // Copy the data.
            if (image_->isValid())
            {
                tensor_helper::copy_to_bitmap(data, image_->accessPixels(), image_->getScanWidth());

                return true;
            }

            return false;
        }

        /// <summary>
        /// Create a floating tensor.
        /// </summary>
        /// <param name="type">The destintation bitmap type</param>
        /// <returns>The result</returns>
        std::optional<memory::tensor<float>> tensor_builder_free_image_impl::to_tensor_float(tensor_layout layout)
        {
            return to_tensor_core<tensor_layout, float, false>(layout, float_converters_);
        }

        /// <summary>
        /// Create a uint8 tensor.
        /// </summary>
        /// <param name="type">The destination bitmap type</param>
        /// <returns>The result</returns>
        std::optional<memory::tensor<uint8_t>> tensor_builder_free_image_impl::to_tensor_uint8(tensor_layout layout)
        {
            return to_tensor_core<tensor_layout, uint8_t, false>(layout, uint8_converters_);
        }

        /// <summary>
        /// Update the parameters of the image.
        /// </summary>
        void tensor_builder_free_image_impl::update_image_parameters_core()
        {
            width_ = image_->width();
            height_ = image_->height();
            stride_ = image_->stride();
            channels_ = image_->channels();
        }

        /// <summary>
        /// Create a shared floating-point tensor.
        /// </summary>
        /// <param name="type">The destintation bitmap type</param>
        /// <returns>The result</returns>
        std::shared_ptr<memory::tensor<float>> tensor_builder_free_image_impl::to_tensor_float_shared(tensor_layout layout)
        {
            return to_tensor_core<tensor_layout, float, true>(layout, float_converters_);
        }

        /// <summary>
        /// Create a shared uint8 tensor.
        /// </summary>
        /// <param name="type">The destintation bitmap type</param>
        /// <returns>The result</returns>
        std::shared_ptr<memory::tensor<uint8_t>> tensor_builder_free_image_impl::to_tensor_uint8_shared(tensor_layout layout)
        {
            return to_tensor_core<tensor_layout, uint8_t, true>(layout, uint8_converters_);
        }

        /// <summary>
        /// Convert a bitmap to a floating image type.
        /// Each pixel value is in the range [-1, 1].
        /// </summary>
        /// <param name="image">The image</param>
        /// <returns>True on success</returns>
        bool tensor_builder_free_image_impl::convert_to_float_range_minus_one_to_one_core(fipImage& image)
        {
            // We convert the image to a 8-bit grayscale one first.
            if (!image.convertToGrayscale())
            {
                return false;
            }

            auto data = image.accessPixels();

            return false;
        }

        /// <summary>
        /// Convert the image to a tensor.
        /// </summary>
        /// <param name="type">The destintion bitmap type</param>
        /// <param name="converters">The converter cache</param>
        /// <returns>The result</returns>
        template<typename TEnum, typename UnderlyingType, bool Shared>
        auto tensor_builder_free_image_impl::to_tensor_core(TEnum type, bitmap_converter_map<TEnum>& converters)
            -> std::conditional_t<Shared, std::shared_ptr<memory::tensor<UnderlyingType>>, std::optional<memory::tensor<UnderlyingType>>>
        {
            static_assert(std::is_enum_v<TEnum>, "The TEnum must be an enumeration type.");
            static_assert(std::is_integral_v<UnderlyingType> || std::is_floating_point_v<UnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");

            auto return_null_wrapper = []
            {
                if constexpr (Shared)
                {
                    return nullptr;
                }
                else
                {
                    return std::nullopt;
                }
            };

            if (!image_->isValid())
            {
                return return_null_wrapper();
            }

            // Find the appropriate converter.
            auto item = converters.find(type);
            if (item == converters.end())
            {
                return return_null_wrapper();
            }

            // Convert the bitmap.
            auto result = item->second(*image_);
            if (!result)
            {
                return return_null_wrapper();
            }

            return tensor_helper::create<UnderlyingType, Shared>(result->accessPixels(), order_, device_, result->width(), result->height(), result->stride(), result->channels());
        }
    }
}
