/// @file   ColorFormat.h
///
/// @brief  Enumeration type for the various color formats in use
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

namespace SE::Core
{

    /// @brief Enumeration type for the various color formats in use
    enum class color_format_t : uint32_t
    {
        UNDEFINED           = 0,
        R32_FLOAT           = 1,
        RG32_FLOAT          = 2,
        RGB32_FLOAT         = 3,
        RGBA32_FLOAT        = 4,
        R16_FLOAT           = 5,
        RG16_FLOAT          = 6,
        RGB16_FLOAT         = 7,
        RGBA16_FLOAT        = 8,
        R8_UNORM            = 9,
        RG8_UNORM           = 10,
        RGB8_UNORM          = 11,
        RGBA8_UNORM         = 12,
        D16_UNORM           = 13,
        X8_D24_UNORM_PACK32 = 14,
        D32_SFLOAT          = 15,
        S8_UINT             = 16,
        D16_UNORM_S8_UINT   = 17,
        D24_UNORM_S8_UINT   = 18,
        D32_UNORM_S8_UINT   = 19,
        BGR8_SRGB           = 20,
        BGRA8_SRGB          = 21
    };

    color_format_t ToLtseFormat( VkFormat C );
    VkFormat       ToVkFormat( color_format_t C );
    uint8_t        GetPixelSize( color_format_t aColorFormat );

} // namespace SE::Core
