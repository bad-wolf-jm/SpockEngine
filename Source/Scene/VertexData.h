#pragma once

#include "Core/Math/Types.h"
// #include "Core/Types.h"

#ifndef __CUDACC__
#    include "Core/GraphicContext//GraphicsPipeline.h"
#endif

namespace LTSE::Core
{
    using namespace math;

#ifndef __CUDACC__
    using namespace LTSE::Graphics;
#endif

    struct EmptyVertexData
    {
#ifndef __CUDACC__
        static LTSE::Graphics::sBufferLayout GetDefaultLayout() { return {}; }
#endif
    };

    struct VertexData
    {
        vec3 Position    = { 0.0f, 0.0f, 0.0f };
        vec3 Normal      = { 0.0f, 0.0f, 0.0f };
        vec2 TexCoords_0 = { 0.0f, 0.0f };
        vec2 TexCoords_1 = { 0.0f, 0.0f };
        vec4 Bones       = { 0.0f, 0.0f, 0.0f, 0.0f };
        vec4 Weights     = { 0.0f, 0.0f, 0.0f, 0.0f };

#ifndef __CUDACC__
        // clang-format off
        static LTSE::Graphics::sBufferLayout GetDefaultLayout()
        {
            return {
                { "Position",   eBufferDataType::VEC3, 0, 0 },
                { "Normal",     eBufferDataType::VEC3, 0, 1 },
                { "TexCoord_0", eBufferDataType::VEC2, 0, 2 },
                { "TexCoord_1", eBufferDataType::VEC2, 0, 3 },
                { "Bones",      eBufferDataType::VEC4, 0, 4 },
                { "Weights",    eBufferDataType::VEC4, 0, 5 } };
        }
        // clang-format on
#endif
    };

    struct SimpleVertexData
    {
        vec3 Position = { 0.0f, 0.0f, 0.0f };
        vec3 Normal   = { 0.0f, 0.0f, 0.0f };

#ifndef __CUDACC__
        // clang-format off
        static LTSE::Graphics::sBufferLayout GetDefaultLayout()
        {
            return {
                { "Position", eBufferDataType::VEC3, 0, 0 },
                { "Normal",   eBufferDataType::VEC3, 0, 1 }
            };
        }
        // clang-format on
#endif
    };

    struct PositionData
    {
        vec3 Position = { 0.0f, 0.0f, 0.0f };

#ifndef __CUDACC__
        // clang-format off
        static LTSE::Graphics::sBufferLayout GetDefaultLayout()
        {
            return {
                { "Position", eBufferDataType::VEC3, 0, 0 }
            };
        }
        // clang-format on
#endif
    };

    struct PositionAndColorData
    {
        vec3 Position = { 0.0f, 0.0f, 0.0f };
        vec4 Color    = { 0.0f, 0.0f, 0.0f, 0.0f };

#ifndef __CUDACC__
        // clang-format off
        static LTSE::Graphics::sBufferLayout GetDefaultLayout()
        {
            return {
                { "Position", eBufferDataType::VEC3, 0, 0 },
                { "Color", eBufferDataType::VEC4, 0, 1 }
            };
        }
        // clang-format on
#endif
    };

} // namespace LTSE::Core
