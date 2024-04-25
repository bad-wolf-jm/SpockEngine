#pragma once

#include "Core/Cuda/Cuda.h"

namespace numlua::linalg
{
    using length_t = size_t;

    template <size_t dimension, typename _Ty>
    struct vect
    {
    };

    template <size_t Rows, size_t Columns, typename _Ty>
    struct matrix
    {
    };

    namespace detail
    {
        template <template <length_t L, typename T> class vec, length_t L,typename R, typename T>
        struct functor1
        {
        };
        
        template <template <length_t L, typename T> class vec, length_t L, typename T>
        struct functor2
        {
        };
    } // namespace detail
} // namespace numlua::linalg
