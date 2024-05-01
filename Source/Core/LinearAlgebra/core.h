#pragma once

#include "Core/Cuda/Cuda.h"

#define LINALG_FUNCTION SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr

namespace numlua::linalg
{
    using length_t = size_t;

    template <size_t dimension, typename _Ty>
    struct vect
    {
    };

    template <size_t _Rows, size_t _Columns, typename _Ty>
    struct matrix_algebra
    {
    };

    template <size_t _Rows, size_t _Columns, typename _Ty>
    struct matrix
    {
    };

    namespace detail
    {
        template <template <length_t L, typename T> class vec, length_t L, typename R, typename T>
        struct functor1
        {
        };

        template <template <length_t L, typename T> class vec, length_t L, typename T>
        struct functor2
        {
        };
    } // namespace detail
} // namespace numlua::linalg
