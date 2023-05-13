#include "ColorFormat.h"

namespace SE::Core
{
    eColorFormat ToLtseFormat( VkFormat C )
    {
        switch( C )
        {
        case VK_FORMAT_R32_SFLOAT: return eColorFormat::R32_FLOAT;
        case VK_FORMAT_R32G32_SFLOAT: return eColorFormat::RG32_FLOAT;
        case VK_FORMAT_R32G32B32_SFLOAT: return eColorFormat::RGB32_FLOAT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return eColorFormat::RGBA32_FLOAT;
        case VK_FORMAT_R16_SFLOAT: return eColorFormat::R16_FLOAT;
        case VK_FORMAT_R16G16_SFLOAT: return eColorFormat::RG16_FLOAT;
        case VK_FORMAT_R16G16B16_SFLOAT: return eColorFormat::RGB16_FLOAT;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return eColorFormat::RGBA16_FLOAT;
        case VK_FORMAT_R8_UNORM: return eColorFormat::R8_UNORM;
        case VK_FORMAT_R8G8_UNORM: return eColorFormat::RG8_UNORM;
        case VK_FORMAT_R8G8B8_UNORM: return eColorFormat::RGB8_UNORM;
        case VK_FORMAT_R8G8B8A8_UNORM: return eColorFormat::RGBA8_UNORM;
        case VK_FORMAT_D16_UNORM: return eColorFormat::D16_UNORM;
        case VK_FORMAT_X8_D24_UNORM_PACK32: return eColorFormat::X8_D24_UNORM_PACK32;
        case VK_FORMAT_D32_SFLOAT: return eColorFormat::D32_SFLOAT;
        case VK_FORMAT_S8_UINT: return eColorFormat::S8_UINT;
        case VK_FORMAT_D16_UNORM_S8_UINT: return eColorFormat::D16_UNORM_S8_UINT;
        case VK_FORMAT_D24_UNORM_S8_UINT: return eColorFormat::D24_UNORM_S8_UINT;
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return eColorFormat::D32_UNORM_S8_UINT;
        case VK_FORMAT_B8G8R8_SRGB: return eColorFormat::BGR8_SRGB;
        case VK_FORMAT_B8G8R8A8_SRGB: return eColorFormat::BGRA8_SRGB;
        }
        return eColorFormat::UNDEFINED;
    }

    VkFormat ToVkFormat( eColorFormat C )
    {
        switch( C )
        {
        case eColorFormat::R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case eColorFormat::RG32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case eColorFormat::RGB32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        case eColorFormat::RGBA32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case eColorFormat::R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
        case eColorFormat::RG16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
        case eColorFormat::RGB16_FLOAT: return VK_FORMAT_R16G16B16_SFLOAT;
        case eColorFormat::RGBA16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case eColorFormat::R8_UNORM: return VK_FORMAT_R8_UNORM;
        case eColorFormat::RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
        case eColorFormat::RGB8_UNORM: return VK_FORMAT_R8G8B8_UNORM;
        case eColorFormat::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case eColorFormat::D16_UNORM: return VK_FORMAT_D16_UNORM;
        case eColorFormat::X8_D24_UNORM_PACK32: return VK_FORMAT_X8_D24_UNORM_PACK32;
        case eColorFormat::D32_SFLOAT: return VK_FORMAT_D32_SFLOAT;
        case eColorFormat::S8_UINT: return VK_FORMAT_S8_UINT;
        case eColorFormat::D16_UNORM_S8_UINT: return VK_FORMAT_D16_UNORM_S8_UINT;
        case eColorFormat::D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
        case eColorFormat::D32_UNORM_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
        case eColorFormat::BGR8_SRGB: return VK_FORMAT_B8G8R8_SRGB;
        case eColorFormat::BGRA8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        }
        return VK_FORMAT_UNDEFINED;
    }

} // namespace SE::Core
