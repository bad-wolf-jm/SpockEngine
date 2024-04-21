#pragma once

#include <cstdint> 

namespace numlua::linalg
{
    template <typename _Ty>
    struct vec2_type
    {
        using value_type = _Ty;

        union
        {
            // clang-format off
            struct { _Ty x, y; };
            struct { _Ty r, g; };
            struct { _Ty s, t; };
            // clang-format on
        };
    };

    using int2 = vec2_type<int32_t>;
    using float2 = vec2_type<float>;
} // namespace numlua::linalg
