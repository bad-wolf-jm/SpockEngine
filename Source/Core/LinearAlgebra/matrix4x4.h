/// @ref core
/// @file glm/detail/type_mat4x4.hpp

#pragma once

#include "core.h"
#include "vector4.h"
#include <cstddef>
#include <limits>

namespace numlua::linalg
{
    template <typename T, qualifier Q>
    struct matrix<4, 4, T>
    {
        typedef vect<4, T>      col_type;
        typedef vect<4, T>      row_type;
        typedef matrix<4, 4, T> type;
        typedef matrix<4, 4, T> transpose_type;
        typedef T               value_type;
        typedef size_t          length_type;

      private:
        col_type value[4];

      public:
        // -- Accesses --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF static constexpr length_type length()
        {
            return 4;
        }

        LINALG_FUNCTION col_type       &operator[]( length_type i ) noexcept;
        LINALG_FUNCTION col_type const &operator[]( length_type i ) const noexcept;

        // -- Constructors --

        LINALG_FUNCTION matrix() = default;
        constexpr matrix( matrix<4, 4, T> const &m );

        constexpr matrix( T s );
        constexpr matrix( T const &x0, T const &y0, T const &z0, T const &w0, T const &x1, T const &y1, T const &z1, T const &w1,
                          T const &x2, T const &y2, T const &z2, T const &w2, T const &x3, T const &y3, T const &z3, T const &w3 );
        constexpr matrix( col_type const &v0, col_type const &v1, col_type const &v2, col_type const &v3 );

        // -- Conversions --

        template <typename X1, typename Y1, typename Z1, typename W1, typename X2, typename Y2, typename Z2, typename W2, typename X3,
                  typename Y3, typename Z3, typename W3, typename X4, typename Y4, typename Z4, typename W4>
        constexpr matrix( X1 const &x1, Y1 const &y1, Z1 const &z1, W1 const &w1, X2 const &x2, Y2 const &y2, Z2 const &z2,
                          W2 const &w2, X3 const &x3, Y3 const &y3, Z3 const &z3, W3 const &w3, X4 const &x4, Y4 const &y4,
                          Z4 const &z4, W4 const &w4 );

        template <typename V1, typename V2, typename V3, typename V4>
        constexpr matrix( vect<4, V1> const &v1, vect<4, V2> const &v2, vect<4, V3> const &v3, vect<4, V4> const &v4 );

        // -- Matrix conversions --

        template <typename U>
        constexpr matrix( matrix<4, 4, U> const &m );

        constexpr matrix( matrix<2, 2, T> const &x );
        constexpr matrix( matrix<3, 3, T> const &x );
        constexpr matrix( matrix<2, 3, T> const &x );
        constexpr matrix( matrix<3, 2, T> const &x );
        constexpr matrix( matrix<2, 4, T> const &x );
        constexpr matrix( matrix<4, 2, T> const &x );
        constexpr matrix( matrix<3, 4, T> const &x );
        constexpr matrix( matrix<4, 3, T> const &x );

        // -- Unary arithmetic operators --

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator=( matrix_type<U> const &m )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] = m[i];

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator+=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] += s;

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator+=( matrix_type<U> const &m )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] += m[i];

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator-=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] -= s;

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator-=( matrix_type<U> const &m )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] -= m[i];

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator*=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] *= s;

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator*=( matrix_type<U> const &m )
        {
            return ( *this = *this * m );
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator/=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] /= s;

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix_type<T> &operator/=( matrix_type<U> const &m )
        {
            return *this *= inverse( m );
        }
    };

    // // -- Unary operators --

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator+( matrix<4, 4, T> const &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator-( matrix<4, 4, T> const &m );

    // // -- Binary operators --

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator+( matrix<4, 4, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator+( T scalar, matrix<4, 4, T> const &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator+( matrix<4, 4, T> const &m1, matrix<4, 4, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator-( matrix<4, 4, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator-( T scalar, matrix<4, 4, T> const &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator-( matrix<4, 4, T> const &m1, matrix<4, 4, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator*( matrix<4, 4, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator*( T scalar, matrix<4, 4, T> const &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION typename matrix<4, 4, T>::col_type operator*( matrix<4, 4, T> const                    &m,
    //                                                               typename matrix<4, 4, T>::row_type const &v );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION typename matrix<4, 4, T>::row_type operator*( typename matrix<4, 4, T>::col_type const &v,
    //                                                               matrix<4, 4, T> const                    &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 4, T> operator*( matrix<4, 4, T> const &m1, matrix<2, 4, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<3, 4, T> operator*( matrix<4, 4, T> const &m1, matrix<3, 4, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator*( matrix<4, 4, T> const &m1, matrix<4, 4, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator/( matrix<4, 4, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator/( T scalar, matrix<4, 4, T> const &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION typename matrix<4, 4, T>::col_type operator/( matrix<4, 4, T> const                    &m,
    //                                                               typename matrix<4, 4, T>::row_type const &v );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION typename matrix<4, 4, T>::row_type operator/( typename matrix<4, 4, T>::col_type const &v,
    //                                                               matrix<4, 4, T> const                    &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 4, T> operator/( matrix<4, 4, T> const &m1, matrix<4, 4, T> const &m2 );

    // // -- Boolean operators --

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION bool operator==( matrix<4, 4, T> const &m1, matrix<4, 4, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION bool operator!=( matrix<4, 4, T> const &m1, matrix<4, 4, T> const &m2 );
} // namespace numlua::linalg

// #ifndef GLM_EXTERNAL_TEMPLATE
// #    include "type_mat4x4.inl"
// #endif // GLM_EXTERNAL_TEMPLATE
