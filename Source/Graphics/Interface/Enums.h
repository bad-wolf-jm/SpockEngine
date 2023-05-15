#pragma once

#include "Core/Types.h"

namespace SE::Graphics
{
    enum class eDescriptorType : uint32_t
    {
        SAMPLER                = 0,
        COMBINED_IMAGE_SAMPLER = 1,
        SAMPLED_IMAGE          = 2,
        STORAGE_IMAGE          = 3,
        UNIFORM_TEXEL_BUFFER   = 4,
        STORAGE_TEXEL_BUFFER   = 5,
        UNIFORM_BUFFER         = 6,
        STORAGE_BUFFER         = 7,
        UNIFORM_BUFFER_DYNAMIC = 8,
        STORAGE_BUFFER_DYNAMIC = 9,
        INPUT_ATTACHMENT       = 10
    };

    enum class eShaderStageTypeFlags : uint32_t
    {
        VERTEX                 = 1,
        GEOMETRY               = 2,
        FRAGMENT               = 4,
        TESSELATION_CONTROL    = 8,
        TESSELATION_EVALUATION = 16,
        COMPUTE                = 32,
        DEFAULT                = 0xFFFFFFFF
    };

    using ShaderStageType = EnumSet<eShaderStageTypeFlags, 0x000001ff>;

    enum class eBufferDataType : uint32_t
    {
        UINT8  = 0,
        UINT16 = 1,
        UINT32 = 2,
        INT8   = 3,
        INT16  = 4,
        INT32  = 5,
        FLOAT  = 6,
        COLOR  = 7,
        VEC2   = 8,
        VEC3   = 9,
        VEC4   = 10,
        IVEC2  = 11,
        IVEC3  = 12,
        IVEC4  = 13,
        UVEC2  = 14,
        UVEC3  = 15,
        UVEC4  = 16
    };

    enum class eDepthCompareOperation : uint32_t
    {
        NEVER            = 0,
        LESS             = 1,
        EQUAL            = 2,
        LESS_OR_EQUAL    = 3,
        GREATER          = 4,
        NOT_EQUAL        = 5,
        GREATER_OR_EQUAL = 6,
        ALWAYS           = 7
    };

    enum class eBlendFactor : uint32_t
    {
        ZERO                     = 0,
        ONE                      = 1,
        SRC_COLOR                = 2,
        ONE_MINUS_SRC_COLOR      = 3,
        DST_COLOR                = 4,
        ONE_MINUS_DST_COLOR      = 5,
        SRC_ALPHA                = 6,
        ONE_MINUS_SRC_ALPHA      = 7,
        DST_ALPHA                = 8,
        ONE_MINUS_DST_ALPHA      = 9,
        CONSTANT_COLOR           = 10,
        ONE_MINUS_CONSTANT_COLOR = 11,
        CONSTANT_ALPHA           = 12,
        ONE_MINUS_CONSTANT_ALPHA = 13,
        SRC_ALPHA_SATURATE       = 14,
        SRC1_COLOR               = 15,
        ONE_MINUS_SRC1_COLOR     = 16,
        SRC1_ALPHA               = 17,
        ONE_MINUS_SRC1_ALPHA     = 18
    };

    enum class eBlendOperation : uint32_t
    {
        ADD              = 0,
        SUBTRACT         = 1,
        REVERSE_SUBTRACT = 2,
        MIN              = 3,
        MAX              = 4
    };

    enum class ePrimitiveTopology : uint32_t
    {
        POINTS    = 0,
        TRIANGLES = 1,
        LINES     = 2
    };

    enum class eFaceCulling : uint32_t
    {
        NONE           = 0,
        FRONT          = 1,
        BACK           = 2,
        FRONT_AND_BACK = 3
    };

} // namespace SE::Graphics