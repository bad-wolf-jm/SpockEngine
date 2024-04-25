/// @ref core
/// @file glm/detail/type_mat4x2.hpp

#pragma once

#include "core.h"
#include "vector2.h"
#include "vector3.h"
#include <cstddef>
#include <limits>

namespace glm
{
    template <typename T, qualifier Q>
    struct matrix<4, 2, T>
    {
        typedef vect<2, T>      col_type;
        typedef vect<4, T>      row_type;
        typedef matrix<4, 2, T> type;
        typedef matrix<2, 4, T> transpose_type;
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

        GLM_DEFAULTED_DEFAULT_CTOR_DECL constexpr matrix() GLM_DEFAULT_CTOR;
        template <qualifier P>
        constexpr matrix( matrix<4, 2, T, P> const &m );

        constexpr matrix( T scalar );
        constexpr matrix( T x0, T y0, T x1, T y1, T x2, T y2, T x3, T y3 );
        constexpr matrix( col_type const &v0, col_type const &v1, col_type const &v2, col_type const &v3 );

        // -- Conversions --

        template <typename X0, typename Y0, typename X1, typename Y1, typename X2, typename Y2, typename X3, typename Y3>
        constexpr matrix( X0 x0, Y0 y0, X1 x1, Y1 y1, X2 x2, Y2 y2, X3 x3, Y3 y3 );

        template <typename V1, typename V2, typename V3, typename V4>
        constexpr matrix( vect<2, V1> const &v1, vect<2, V2> const &v2, vect<2, V3> const &v3, vect<2, V4> const &v4 );

        // -- Matrix conversions --

        template <typename U, qualifier P>
        constexpr matrix( matrix<4, 2, U, P> const &m );

        constexpr matrix( matrix<2, 2, T> const &x );
        constexpr matrix( matrix<3, 3, T> const &x );
        constexpr matrix( matrix<4, 4, T> const &x );
        constexpr matrix( matrix<2, 3, T> const &x );
        constexpr matrix( matrix<3, 2, T> const &x );
        constexpr matrix( matrix<2, 4, T> const &x );
        constexpr matrix( matrix<4, 3, T> const &x );
        constexpr matrix( matrix<3, 4, T> const &x );

        // -- Unary arithmetic operators --

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator=( matrix<4, 2, U> const &m );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator+=( U s );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator+=( matrix<4, 2, U> const &m );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator-=( U s );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator-=( matrix<4, 2, U> const &m );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator*=( U s );
        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator/=( U s );

        // -- Increment and decrement operators --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator++();
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> &operator--();
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T>  operator++( int );
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T>  operator--( int );
    };

    // -- Unary operators --

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator+( matrix<4, 2, T> const &m );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator-( matrix<4, 2, T> const &m );

    // -- Binary operators --

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator+( matrix<4, 2, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator+( matrix<4, 2, T> const &m1, matrix<4, 2, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator-( matrix<4, 2, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator-( matrix<4, 2, T> const &m1, matrix<4, 2, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator*( matrix<4, 2, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator*( T scalar, matrix<4, 2, T> const &m );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<4, 2, T>::col_type
    operator*( matrix<4, 2, T> const &m, typename matrix<4, 2, T>::row_type const &v );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<4, 2, T>::row_type
    operator*( typename matrix<4, 2, T>::col_type const &v, matrix<4, 2, T> const &m );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator*( matrix<4, 2, T> const &m1, matrix<2, 4, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<3, 2, T> operator*( matrix<4, 2, T> const &m1, matrix<3, 4, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator*( matrix<4, 2, T> const &m1, matrix<4, 4, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator/( matrix<4, 2, T> const &m, T scalar );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator/( T scalar, matrix<4, 2, T> const &m );

    // -- Boolean operators --

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator==( matrix<4, 2, T> const &m1, matrix<4, 2, T> const &m2 );

    template <typename T, qualifier Q>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator!=( matrix<4, 2, T> const &m1, matrix<4, 2, T> const &m2 );
} // namespace glm

#ifndef GLM_EXTERNAL_TEMPLATE
#    include "type_mat4x2.inl"
#endif
