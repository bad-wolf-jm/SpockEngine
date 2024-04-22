
#pragma once
#include "core.h"
namespace numlua::linalg
{
    template <typename _Ty>
    struct vect<3, _Ty>
    {
        using value_type = _Ty;

        union
        {
            // clang-format off
            struct { _Ty x, y, z; };
            struct { _Ty r, g, b; };
            struct { _Ty s, t, p; };
            // clang-format on
        };
    };
} // namespace numlua::linalg
