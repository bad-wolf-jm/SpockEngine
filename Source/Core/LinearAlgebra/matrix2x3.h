/// @ref core
/// @file glm/detail/type_mat2x3.hpp

#pragma once

#include "core.h"
#include "vector2.h"
#include "vector3.h"
#include <cstddef>
#include <limits>

namespace numlua::linalg
{
    template <typename T, qualifier Q>
    struct matrix<2, 3, T>
    {
        typedef vect<3, T>      col_type;
        typedef vect<2, T>      row_type;
        typedef matrix<2, 3, T> type;
        typedef matrix<3, 2, T> transpose_type;
        typedef T               value_type;
        typedef length_t        length_type;

      private:
        col_type value[2];

      public:
        // -- Accesses --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF static constexpr length_type length()
        {
            return 2;
        }

        LINALG_FUNCTION col_type &operator[]( length_type i ) noexcept
        {
            return this->value[i];
        }

        LINALG_FUNCTION col_type const &operator[]( length_type i ) const noexcept
        {
            return this->value[i];
        }

        // -- Constructors --

        LINALG_FUNCTION matrix()
            : value{ col_type( 1, 0 ), col_type( 0, 1 ), col_type( 0, 0 ) }
        {
        }

        constexpr matrix( matrix<2, 3, T> const &m )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( m[2] ) }
        {
        }

        constexpr matrix( T scalar )
            : value{ col_type( s, 0 ), col_type( 0, s ), col_type( 0, 0 ) }
        {
        }
        constexpr matrix( T x0, T y0, T x1, T y1, T x2, T y2 )
            : value{ col_type( x0, y0 ), col_type( x1, y1 ), col_type( x2, y2 ) }
        {
        }
        constexpr matrix( col_type const &v0, col_type const &v1, col_type const &v2 )
            : value{ col_type( v0 ), col_type( v1 ), col_type( v2 ) }
        {
        }

        // -- Conversions --

        template <typename X1, typename Y1, typename Z1, typename X2, typename Y2, typename Z2>
        constexpr matrix( X0 x0, Y0 y0, X1 x1, Y1 y1, X2 x2, Y2 y2 )
            : value{ col_type( x0, y0 ), col_type( x1, y1 ), col_type( x2, y2 ) }
        {
        }

        template <typename U, typename V>
        constexpr matrix( vect<2, V0> const &v0, vect<2, V1> const &v1, vect<2, V2> const &v2 )
            : value{ col_type( v0 ), col_type( v1 ), col_type( v2 ) }
        {
        }

        // -- Matrix conversions --

        template <typename U>
        constexpr matrix( matrix<2, 3, U> const &m )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( m[2] ) }
        {
        }

        constexpr matrix( matrix<2, 2, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( 0 ) }
        {
        }

        constexpr matrix( matrix<3, 3, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( m[2] ) }
        {
        }

        constexpr matrix( matrix<4, 4, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( m[2] ) }
        {
        }

        constexpr matrix( matrix<2, 4, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( m[2] ) }
        {
        }

        constexpr matrix( matrix<3, 2, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( 0 ) }
        {
        }

        constexpr matrix( matrix<3, 4, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( m[2] ) }
        {
        }

        constexpr matrix( matrix<4, 2, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( 0 ) }
        {
        }

        constexpr matrix( matrix<4, 3, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ), col_type( m[2] ) }
        {
        }

        // -- Unary arithmetic operators --

        template <typename U>
        LINALG_FUNCTION matrix<2, 3, T> &operator=( matrix<2, 3, U> const &m )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] = m[i];

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix<2, 3, T> &operator+=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] += s;

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix<2, 3, T> &operator+=( matrix<2, 3, U> const &m )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] += m[i];

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix<2, 3, T> &operator-=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] -= s;

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix<2, 3, T> &operator-=( matrix<2, 3, U> const &m )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] -= m[i];

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION matrix<2, 3, T> &operator*=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] *= s;

            return *this;
        }
        
        template <typename U>
        LINALG_FUNCTION matrix<2, 3, T> &operator/=( U s )
        {
            for( int i = 0; i < length(); i++ )
                this->value[i] /= s;

            return *this;
        }
    };

    // // -- Unary operators --

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator+( matrix<2, 3, T> const &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator-( matrix<2, 3, T> const &m );

    // // -- Binary operators --

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator+( matrix<2, 3, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator+( matrix<2, 3, T> const &m1, matrix<2, 3, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator-( matrix<2, 3, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator-( matrix<2, 3, T> const &m1, matrix<2, 3, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator*( matrix<2, 3, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator*( T scalar, matrix<2, 3, T> const &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION typename matrix<2, 3, T>::col_type operator*( matrix<2, 3, T> const                    &m,
    //                                                               typename matrix<2, 3, T>::row_type const &v );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION typename matrix<2, 3, T>::row_type operator*( typename matrix<2, 3, T>::col_type const &v,
    //                                                               matrix<2, 3, T> const                    &m );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator*( matrix<2, 3, T> const &m1, matrix<2, 2, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<3, 3, T> operator*( matrix<2, 3, T> const &m1, matrix<3, 2, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<4, 3, T> operator*( matrix<2, 3, T> const &m1, matrix<4, 2, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator/( matrix<2, 3, T> const &m, T scalar );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION matrix<2, 3, T> operator/( T scalar, matrix<2, 3, T> const &m );

    // // -- Boolean operators --

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION bool operator==( matrix<2, 3, T> const &m1, matrix<2, 3, T> const &m2 );

    // template <typename T, qualifier Q>
    // LINALG_FUNCTION bool operator!=( matrix<2, 3, T> const &m1, matrix<2, 3, T> const &m2 );
} // namespace numlua::linalg

// #ifndef GLM_EXTERNAL_TEMPLATE
// #    include "type_mat2x3.inl"
// #endif
