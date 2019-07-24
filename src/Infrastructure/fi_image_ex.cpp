#include "fi_image_ex.hpp"

namespace glasssix
{
    namespace excalibur
    {
        bool fi_image_ex::convert_to_4bits(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertTo4Bits, other);
        }

        bool fi_image_ex::convert_to_8bits(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertTo8Bits, other);
        }

        bool fi_image_ex::convert_to_grayscale(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertToGreyscale, other);
        }

        bool fi_image_ex::convert_to_16bits_555(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertTo16Bits555, other);
        }

        bool fi_image_ex::convert_to_16bits_565(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertTo16Bits565, other);
        }

        bool fi_image_ex::convert_to_24bits(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertTo24Bits, other);
        }

        bool fi_image_ex::convert_to_32bits(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertTo32Bits, other);
        }

        bool fi_image_ex::convert_to_float(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertToFloat, other);
        }

        bool fi_image_ex::convert_to_rgbf(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertToRGBF, other);
        }

        bool fi_image_ex::convert_to_rgbaf(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertToRGBAF, other);
        }

        bool fi_image_ex::convert_to_uint16(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertToUINT16, other);
        }

        bool fi_image_ex::convert_to_rgb16(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertToRGB16, other);
        }

        bool fi_image_ex::convert_to_rgba16(fipImage& other)
        {
            return convert_to_core(FreeImage_ConvertToRGBA16, other);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_4bits()
        {
            return convert_to_core(FreeImage_ConvertTo4Bits);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_8bits()
        {
            return convert_to_core(FreeImage_ConvertTo8Bits);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_grayscale()
        {
            return convert_to_core(FreeImage_ConvertToGreyscale);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_16bits_555()
        {
            return convert_to_core(FreeImage_ConvertTo16Bits555);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_16bits_565()
        {
            return convert_to_core(FreeImage_ConvertTo16Bits565);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_24bits()
        {
            return convert_to_core(FreeImage_ConvertTo24Bits);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_32bits()
        {
            return convert_to_core(FreeImage_ConvertTo32Bits);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_float()
        {
            return convert_to_core(FreeImage_ConvertToFloat);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_rgbf()
        {
            return convert_to_core(FreeImage_ConvertToRGBF);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_rgbaf()
        {
            return convert_to_core(FreeImage_ConvertToRGBAF);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_uint16()
        {
            return convert_to_core(FreeImage_ConvertToUINT16);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_rgb16()
        {
            return convert_to_core(FreeImage_ConvertToRGB16);
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_rgba16()
        {
            return convert_to_core(FreeImage_ConvertToRGBA16);
        }

        int fi_image_ex::width() const
        {
            return static_cast<int>(getWidth());
        }

        int fi_image_ex::height() const
        {
            return static_cast<int>(getHeight());
        }

        int fi_image_ex::stride() const
        {
            return static_cast<int>(getScanWidth());
        }

        int fi_image_ex::channels() const
        {
            return static_cast<int>(getBitsPerPixel()) / static_cast<int>(channel_byte_mapping_[getImageType()]) / uint8_bits_;
        }

        bool fi_image_ex::convert_to_core(const std::function<FIBITMAP*(FIBITMAP*)>& handler, fipImage& other)
        {
            auto bitmap = handler ? handler(*this) : nullptr;

            if (bitmap != nullptr)
            {
                other = bitmap;

                return true;
            }

            return false;
        }

        std::optional<fi_image_ex> fi_image_ex::convert_to_core(const std::function<FIBITMAP* (FIBITMAP*)>& handler)
        {
            auto bitmap = handler ? handler(*this) : nullptr;

            if (bitmap != nullptr)
            {
                auto result = std::make_optional<fi_image_ex>();
                *result = bitmap;

                return result;
            }

            return std::nullopt;
        }
    }
}
