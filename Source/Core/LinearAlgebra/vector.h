#pragma once

#include "vector2.h"
#include "vector3.h"
#include "vector4.h"

namespace numlua::linalg
{
    using float2 = vect<2, float>;
    using float3 = vect<3, float>;
    using float4 = vect<4, float>;

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator+( vect<L, _Ty> const &v )
    {
        return v;
    }

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator+( vect<L, _Ty> const &v, _Ty scalar )
    {
        return vect<L, _Ty>( v ) += scalar;
    }

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator+( _Ty scalar, vect<L, _Ty> const &v )
    {
        return vect<L, _Ty>( v ) += scalar;
    }

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator+( vect<L, _Ty> const &v1, vect<L, _Ty> const &v2 )
    {
        return vect<L, _Ty>( v1 ) += v2;
    }

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator-( vect<L, _Ty> const &v )
    {
        return vect<L, _Ty>( 0 ) -= v;
    }

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator-( vect<L, _Ty> const &v, _Ty scalar )
    {
        return vect<L, _Ty>( v ) -= scalar;
    }

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator-( _Ty scalar, vect<L, _Ty> const &v )
    {
        return vect<L, _Ty>( scalar ) -= v;
    }

    template <size_t L, typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<L, _Ty> operator-( vect<L, _Ty> const &v1, vect<L, _Ty> const &v2 )
    {
        return vect<L, _Ty>( v1 ) -= v2;
    }
} // namespace numlua::linalg
