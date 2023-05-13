#pragma once

#include "Core/Math/Types.h"

#ifndef __CUDACC__
#    include "Graphics/Vulkan/VkGraphicsPipeline.h"
#endif

#ifndef __CUDACC__
using namespace SE::Graphics;
#endif

struct Particle
{
    math::vec4 PositionAndSize;
    math::vec4 Color;

#ifndef __CUDACC__
    static SE::Graphics::sBufferLayout GetDefaultLayout()
    {
        return { { "Position", eBufferDataType::VEC4, 0, 1 }, { "Color", eBufferDataType::VEC4, 0, 2 } };
    }
#endif
};
