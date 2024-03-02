#include "ColorFormat.h"

namespace SE::Core
{
    color_format_t ToLtseFormat( VkFormat C )
    {
        switch( C )
        {
        case VK_FORMAT_R32_SFLOAT:
            return color_format_t::R32_FLOAT;
        case VK_FORMAT_R32G32_SFLOAT:
            return color_format_t::RG32_FLOAT;
        case VK_FORMAT_R32G32B32_SFLOAT:
            return color_format_t::RGB32_FLOAT;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return color_format_t::RGBA32_FLOAT;
        case VK_FORMAT_R16_SFLOAT:
            return color_format_t::R16_FLOAT;
        case VK_FORMAT_R16G16_SFLOAT:
            return color_format_t::RG16_FLOAT;
        case VK_FORMAT_R16G16B16_SFLOAT:
            return color_format_t::RGB16_FLOAT;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return color_format_t::RGBA16_FLOAT;
        case VK_FORMAT_R8_UNORM:
            return color_format_t::R8_UNORM;
        case VK_FORMAT_R8G8_UNORM:
            return color_format_t::RG8_UNORM;
        case VK_FORMAT_R8G8B8_UNORM:
            return color_format_t::RGB8_UNORM;
        case VK_FORMAT_R8G8B8A8_UNORM:
            return color_format_t::RGBA8_UNORM;
        case VK_FORMAT_D16_UNORM:
            return color_format_t::D16_UNORM;
        case VK_FORMAT_X8_D24_UNORM_PACK32:
            return color_format_t::X8_D24_UNORM_PACK32;
        case VK_FORMAT_D32_SFLOAT:
            return color_format_t::D32_SFLOAT;
        case VK_FORMAT_S8_UINT:
            return color_format_t::S8_UINT;
        case VK_FORMAT_D16_UNORM_S8_UINT:
            return color_format_t::D16_UNORM_S8_UINT;
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return color_format_t::D24_UNORM_S8_UINT;
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return color_format_t::D32_UNORM_S8_UINT;
        case VK_FORMAT_B8G8R8_SRGB:
            return color_format_t::BGR8_SRGB;
        case VK_FORMAT_B8G8R8A8_SRGB:
            return color_format_t::BGRA8_SRGB;
        }
        return color_format_t::UNDEFINED;
    }

    VkFormat ToVkFormat( color_format_t C )
    {
        switch( C )
        {
        case color_format_t::R32_FLOAT:
            return VK_FORMAT_R32_SFLOAT;
        case color_format_t::RG32_FLOAT:
            return VK_FORMAT_R32G32_SFLOAT;
        case color_format_t::RGB32_FLOAT:
            return VK_FORMAT_R32G32B32_SFLOAT;
        case color_format_t::RGBA32_FLOAT:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case color_format_t::R16_FLOAT:
            return VK_FORMAT_R16_SFLOAT;
        case color_format_t::RG16_FLOAT:
            return VK_FORMAT_R16G16_SFLOAT;
        case color_format_t::RGB16_FLOAT:
            return VK_FORMAT_R16G16B16_SFLOAT;
        case color_format_t::RGBA16_FLOAT:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case color_format_t::R8_UNORM:
            return VK_FORMAT_R8_UNORM;
        case color_format_t::RG8_UNORM:
            return VK_FORMAT_R8G8_UNORM;
        case color_format_t::RGB8_UNORM:
            return VK_FORMAT_R8G8B8_UNORM;
        case color_format_t::RGBA8_UNORM:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case color_format_t::D16_UNORM:
            return VK_FORMAT_D16_UNORM;
        case color_format_t::X8_D24_UNORM_PACK32:
            return VK_FORMAT_X8_D24_UNORM_PACK32;
        case color_format_t::D32_SFLOAT:
            return VK_FORMAT_D32_SFLOAT;
        case color_format_t::S8_UINT:
            return VK_FORMAT_S8_UINT;
        case color_format_t::D16_UNORM_S8_UINT:
            return VK_FORMAT_D16_UNORM_S8_UINT;
        case color_format_t::D24_UNORM_S8_UINT:
            return VK_FORMAT_D24_UNORM_S8_UINT;
        case color_format_t::D32_UNORM_S8_UINT:
            return VK_FORMAT_D32_SFLOAT_S8_UINT;
        case color_format_t::BGR8_SRGB:
            return VK_FORMAT_B8G8R8_SRGB;
        case color_format_t::BGRA8_SRGB:
            return VK_FORMAT_B8G8R8A8_SRGB;
        }
        return VK_FORMAT_UNDEFINED;
    }

    uint8_t GetPixelSize( color_format_t aColorFormat )
    {
        switch( aColorFormat )
        {
        case color_format_t::R32_FLOAT:
            return 4;
        case color_format_t::RG32_FLOAT:
            return 8;
        case color_format_t::RGB32_FLOAT:
            return 12;
        case color_format_t::RGBA32_FLOAT:
            return 16;
        case color_format_t::R16_FLOAT:
            return 2;
        case color_format_t::RG16_FLOAT:
            return 4;
        case color_format_t::RGB16_FLOAT:
            return 6;
        case color_format_t::RGBA16_FLOAT:
            return 8;
        case color_format_t::R8_UNORM:
            return 1;
        case color_format_t::RG8_UNORM:
            return 2;
        case color_format_t::RGB8_UNORM:
            return 3;
        case color_format_t::RGBA8_UNORM:
            return 4;
        case color_format_t::D16_UNORM:
            return 2;
        case color_format_t::X8_D24_UNORM_PACK32:
            return 4;
        case color_format_t::D32_SFLOAT:
            return 4;
        case color_format_t::S8_UINT:
            return 1;
        case color_format_t::D16_UNORM_S8_UINT:
            return 3;
        case color_format_t::D24_UNORM_S8_UINT:
            return 4;
        case color_format_t::D32_UNORM_S8_UINT:
            return 5;
        case color_format_t::BGR8_SRGB:
            return 3;
        case color_format_t::BGRA8_SRGB:
            return 4;
        }
        return 0;
    }
} // namespace SE::Core
