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

#include "Graphics/API/ColorFormat.h"

namespace LTSE::Core
{
    eColorFormat ToLtseFormat( VkFormat C );

    VkFormat     ToVkFormat( eColorFormat C );

} // namespace LTSE::Core
