#include "tensor_builder_free_image_impl.hpp"

namespace glasssix
{
    namespace excalibur
    {
        tensor_builder_free_image_impl::tensor_builder_free_image_impl() : device_{ -1 }, order_{ NHWC }, width_{}, height_{}, stride_{}, channels_{}
        {
            image_ = std::make_shared<fipImage>();
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
        void tensor_builder_free_image_impl::tensor_parameters(orderType order)
        {
            order_ = order;
        }

        /// <summary>
        /// Set the parameters for building a tensor.
        /// </summary>
        /// <param name="order">The memory order</param>
        /// <param name="device">The device ID</param>
        void tensor_builder_free_image_impl::tensor_parameters(orderType order, int device)
        {
            order_ = order;
            device_ = device;
        }

        /// <summary>
        /// Create a floating tensor.
        /// </summary>
        /// <param name="type">The destintation bitmap type</param>
        /// <returns>The result</returns>
        std::optional<tensor<float>> tensor_builder_free_image_impl::to_tensor(tensor_float_type type)
        {
            return to_tensor_core<tensor_float_type, float>(type, float_converters_);
        }

        /// <summary>
        /// Create a uint8 tensor.
        /// </summary>
        /// <param name="type">The destination bitmap type</param>
        /// <returns>The result</returns>
        std::optional<tensor<uint8_t>> tensor_builder_free_image_impl::to_tensor(tensor_uint8_type type)
        {
            return to_tensor_core<tensor_uint8_type, uint8_t>(type, uint8_converters_);
        }

        /// <summary>
        /// Update the parameters of the image.
        /// </summary>
        void tensor_builder_free_image_impl::update_image_parameters_core()
        {
            width_ = image_->getWidth();
            height_ = image_->getHeight();
            stride_ = image_->getScanWidth();
            channels_ = image_->getBitsPerPixel() / static_cast<int>(channel_byte_mapping_[image_->getImageType()]) / uint8_bits_;
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
    }
}
