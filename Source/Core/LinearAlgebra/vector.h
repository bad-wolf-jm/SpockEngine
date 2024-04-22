#pragma once

#include "vector2.h"
#include "vector3.h"
#include "vector4.h"

namespace numlua::linalg
{
    using int2   = vect<2, int32_t>;
    using float2 = vect<2, float>;

    using int3   = vect<3, int32_t>;
    using float3 = vect<3, float>;

    using int4    = vect<4, int32_t>;
    using float4 = vect<4, float>;
} // namespace numlua::linalg
