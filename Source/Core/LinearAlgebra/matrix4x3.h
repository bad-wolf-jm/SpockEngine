/// @ref core
/// @file glm/detail/type_mat4x3.hpp

#pragma once

#include "core.h"
#include "vector3.h"
#include "vector4.h"
#include <cstddef>
#include <limits>

namespace glm
{
    template <typename T, qualifier Q>
    struct matrix<4, 3, T>
    {
        typedef vect<3, T>      col_type;
        typedef vect<4, T>      row_type;
        typedef matrix<4, 3, T> type;
        typedef matrix<3, 4, T> transpose_type;
        typedef T               value_type;
        typedef length_t        length_type;

      private:
        col_type value[4];

      public:
        // -- Accesses --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF static constexpr length_type length()
        {
            return 4;
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr col_type       &operator[]( length_type i ) noexcept;
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr col_type const &operator[]( length_type i ) const noexcept;

        // -- Constructors --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix() = default;
        constexpr matrix( matrix<4, 3, T> const &m );

        constexpr matrix( T s );
        constexpr matrix( T const &x0, T const &y0, T const &z0, T const &x1, T const &y1, T const &z1, T const &x2, T const &y2,
                          T const &z2, T const &x3, T const &y3, T const &z3 );
        constexpr matrix( col_type const &v0, col_type const &v1, col_type const &v2, col_type const &v3 );

        // -- Conversions --

        template <typename X1, typename Y1, typename Z1, typename X2, typename Y2, typename Z2, typename X3, typename Y3, typename Z3,
                  typename X4, typename Y4, typename Z4>
        constexpr matrix( X1 const &x1, Y1 const &y1, Z1 const &z1, X2 const &x2, Y2 const &y2, Z2 const &z2, X3 const &x3,
                          Y3 const &y3, Z3 const &z3, X4 const &x4, Y4 const &y4, Z4 const &z4 );

        template <typename V1, typename V2, typename V3, typename V4>
        constexpr matrix( vect<3, V1> const &v1, vect<3, V2> const &v2, vect<3, V3> const &v3, vect<3, V4> const &v4 );

        // -- Matrix conversions --

        template <typename U>
        constexpr matrix( matrix<4, 3, U> const &m );

        constexpr matrix( matrix<2, 2, T> const &x );
        constexpr matrix( matrix<3, 3, T> const &x );
        constexpr matrix( matrix<4, 4, T> const &x );
        constexpr matrix( matrix<2, 3, T> const &x );
        constexpr matrix( matrix<3, 2, T> const &x );
        constexpr matrix( matrix<2, 4, T> const &x );
        constexpr matrix( matrix<4, 2, T> const &x );
        constexpr matrix( matrix<3, 4, T> const &x );

        // -- Unary arithmetic operators --

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator=( matrix<4, 3, U> const &m );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator+=( U s );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator+=( matrix<4, 3, U> const &m );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator-=( U s );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator-=( matrix<4, 3, U> const &m );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator*=( U s );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator/=( U s );

        // -- Increment and decrement operators --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator++();
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> &operator--();
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T>  operator++( int );
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T>  operator--( int );
    };

    // -- Unary operators --

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator+( matrix<4, 3, T> const &m );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator-( matrix<4, 3, T> const &m );

    // -- Binary operators --

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator+( matrix<4, 3, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator+( matrix<4, 3, T> const &m1, matrix<4, 3, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator-( matrix<4, 3, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator-( matrix<4, 3, T> const &m1, matrix<4, 3, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator*( matrix<4, 3, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator*( T scalar, matrix<4, 3, T> const &m );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<4, 3, T>::col_type
    operator*( matrix<4, 3, T> const &m, typename matrix<4, 3, T>::row_type const &v );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<4, 3, T>::row_type
    operator*( typename matrix<4, 3, T>::col_type const &v, matrix<4, 3, T> const &m );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 3, T> operator*( matrix<4, 3, T> const &m1, matrix<2, 4, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<3, 3, T> operator*( matrix<4, 3, T> const &m1, matrix<3, 4, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator*( matrix<4, 3, T> const &m1, matrix<4, 4, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator/( matrix<4, 3, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 3, T> operator/( T scalar, matrix<4, 3, T> const &m );

    // -- Boolean operators --

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator==( matrix<4, 3, T> const &m1, matrix<4, 3, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator!=( matrix<4, 3, T> const &m1, matrix<4, 3, T> const &m2 );
} // namespace glm

#ifndef GLM_EXTERNAL_TEMPLATE
#    include "type_mat4x3.inl"
#endif // GLM_EXTERNAL_TEMPLATE
