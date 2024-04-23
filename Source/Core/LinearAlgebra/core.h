#pragma once

#include "Core/Cuda/Cuda.h"

namespace numlua::linalg
{
    using length_t = size_t;

    template <size_t dimension, typename _Ty>
    struct vect
    {
    };

    namespace details
    {
        template <template <length_t L, typename T> class vec, length_t L, typename T>
        struct functor2
        {
        };

        template <template <length_t L, typename T> class vec, typename T>
        struct functor2<vec, 4, T>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vec<4, T> call( T ( *Func )( T x, T y ), vect<4, T> const &a, vect<4, T> const &b )
            {
                return vec<4, T>( Func( a.x, b.x ), Func( a.y, b.y ), Func( a.z, b.z ), Func( a.w, b.w ) );
            }

            template <class Fct>
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vec<4, T> call( Fct Func, vec<4, T> const &a, vec<4, T> const &b )
            {
                return vec<4, T>( Func( a.x, b.x ), Func( a.y, b.y ), Func( a.z, b.z ), Func( a.w, b.w ) );
            }
        };
    } // namespace details
} // namespace numlua::linalg
