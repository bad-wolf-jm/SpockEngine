#pragma once

#include "vector2.h"
#include "vector3.h"
#include "vector4.h"

namespace numlua::linalg
{
    using int2   = vec2_type<int32_t>;
    using float2 = vec2_type<float>;

    using int3   = vec3_type<int32_t>;
    using float3 = vec3_type<float>;

    using int4    = vec4_type<int32_t>;
    using float4 = vec4_type<float>;
} // namespace numlua::linalg
