#pragma once

#include <string>
#include <vector>

#include "Graphics/API.h"

namespace SE::Graphics
{
    void Compile( eShaderStageTypeFlags aShaderStage, string_t const &aCode, vec_t<uint32_t> &aOutput );
} // namespace SE::Graphics