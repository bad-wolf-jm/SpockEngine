/// @file   ScalarTypes.h
///
/// @brief  Definitions for Abstract scalar types
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert. All rights reserved.

#pragma once

#include "Core/Definitions.h"

using namespace SE::Core;

namespace SE::TensorOps
{
    enum class broadcast_hint_t : uint8_t
    {
        LEFT  = 0,
        RIGHT = 1,
        NONE  = 2
    };
} // namespace SE::TensorOps
