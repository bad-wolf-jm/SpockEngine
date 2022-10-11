#pragma once

#include "Core/Math/Types.h"

#pragma once

#include "Core/Math/Types.h"

#ifndef __CUDACC__
#    include "Developer/GraphicContext/GraphicsPipeline.h"
#endif

#ifndef __CUDACC__
using namespace LTSE::Graphics;
#endif

struct Particle
{
    math::vec4 PositionAndSize;
    math::vec4 Color;

#ifndef __CUDACC__
    static LTSE::Graphics::sBufferLayout GetDefaultLayout() { return { { "Position", eBufferDataType::VEC4, 0, 1 }, { "Color", eBufferDataType::VEC4, 0, 2 } }; }
#endif
};
