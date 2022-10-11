/// @file   ColorFormat.h
///
/// @brief  Enumeration type for the various color formats in use
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

namespace LTSE::Core
{

    /// @brief Enumeration type for the various color formats in use
    enum class eColorFormat : uint32_t
    {
        UNDEFINED   = 0, /**!< specifies a one-component, 32-bit signed floating-point format that has a single 32-bit R component.*/
        R32_FLOAT   = 1, /**!< specifies a one-component, 32-bit signed floating-point format that has a single 32-bit R component.*/
        RG32_FLOAT  = 2, /**!< specifies a two-component, 64-bit signed floating-point format that has a 32-bit R component in bytes 0..3, and a 32-bit G component in bytes 4..7.*/
        RGB32_FLOAT = 3, /**!< specifies a three-component, 96-bit signed floating-point format that has a 32-bit R component in bytes 0..3, a 32-bit G component in bytes 4..7, and
                            a 32-bit B component in bytes 8..11. */
        RGBA32_FLOAT = 4, /**!< specifies a four-component, 128-bit unsigned integer format that has a 32-bit R component in bytes 0..3, a 32-bit G component in bytes 4..7, a
                             32-bit B component in bytes 8..11, and a 32-bit A component in bytes 12..15.*/
        R16_FLOAT   = 5,  /**!< specifies a one-component, 16-bit signed floating-point format that has a single 16-bit R component.*/
        RG16_FLOAT  = 6, /**!< specifies a two-component, 16-bit signed floating-point format that has a 16-bit R component in bytes 0..3, and a 32-bit G component in bytes 4..7.*/
        RGB16_FLOAT = 7, /**!< specifies a three-component, 96-bit signed floating-point format that has a 16-bit R component in bytes 0..3, a 16-bit G component in bytes 4..7, and
                            a 16-bit B component in bytes 8..11. */
        RGBA16_FLOAT = 8, /**!< specifies a four-component, 128-bit unsigned integer format that has a 16-bit R component in bytes 0..3, a 16-bit G component in bytes 4..7, a
                             16-bit B component in bytes 8..11, and a 16-bit A component in bytes 12..15.*/
        R8_UNORM   = 9,   /**!< specifies a one-component, 8-bit unsigned normalized format that has a single 8-bit R component.*/
        RG8_UNORM  = 10,  /**!< specifies a two-component, 16-bit unsigned normalized format that has an 8-bit R component in byte 0, and an 8-bit G component in byte 1.*/
        RGB8_UNORM = 11,  /**!< specifies a three-component, 24-bit unsigned normalized format that has an 8-bit R component in byte 0, an 8-bit G component in byte 1, and an 8-bit
                             B component in byte 2.*/
        RGBA8_UNORM = 12, /**!< specifies a four-component, 32-bit unsigned normalized format that has an 8-bit R component stored with sRGB nonlinear encoding in byte 0, an 8-bit
                             G component stored with sRGB nonlinear encoding in byte 1, an 8-bit B component stored with sRGB nonlinear encoding in byte 2, and an 8-bit A component
                             in byte 3.*/
        D16_UNORM = 13,   /**!< specifies a one-component, 16-bit unsigned normalized format that has a single 16-bit depth component. */
        X8_D24_UNORM_PACK32 =
            14,          /**!< specifies a two-component, 32-bit format that has 24 unsigned normalized bits in the depth component and, optionally:, 8 bits that are unused. */
        D32_SFLOAT = 15, /**!< specifies a one-component, 32-bit signed floating-point format that has 32 bits in the depth component. */
        S8_UINT    = 16, /**!< specifies a one-component, 8-bit unsigned integer format that has 8 bits in the stencil component. */
        D16_UNORM_S8_UINT =
            17, /**!< specifies a two-component, 24-bit format that has 16 unsigned normalized bits in the depth component and 8 unsigned integer bits in the stencil component. */
        D24_UNORM_S8_UINT = 18, /**!< specifies a two-component, 32-bit packed format that has 8 unsigned integer bits in the stencil component, and 24 unsigned normalized bits in
                                   the depth component. */
        D32_UNORM_S8_UINT = 19, /**!< specifies a two-component format that has 32 signed float bits in the depth component and 8 unsigned integer bits in the stencil component.
                                   There are optionally: 24 bits that are unused. */
        BGR8_SRGB = 20,
        BGRA8_SRGB = 21
    };

    eColorFormat ToLtseFormat( VkFormat C );
    VkFormat ToVkFormat( eColorFormat C );

} // namespace LTSE::Core
