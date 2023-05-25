#pragma once

#include <string>
#include <vector>

#include "Graphics/API.h"

namespace SE::Graphics
{
    void Compile( eShaderStageTypeFlags aShaderStage, std::string const &aCode, std::vector<uint32_t> &aOutput );
} // namespace SE::Graphics