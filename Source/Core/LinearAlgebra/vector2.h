#pragma once

#include "core.h"

namespace numlua::linalg
{
    template <typename _Ty>
    struct vect<2, _Ty>
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

} // namespace numlua::linalg
