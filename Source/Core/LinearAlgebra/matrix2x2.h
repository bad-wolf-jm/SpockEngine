/// @ref core
/// @file glm/detail/type_mat2x2.hpp

#pragma once
#include "core.h"
#include "vector2.h"
#include <cstddef>
#include <limits>

namespace glm
{
    template <typename T>
    struct matrix<2, 2, T>
    {
        typedef vect<2, T>      col_type;
        typedef vect<2, T>      row_type;
        typedef matrix<2, 2, T> type;
        typedef matrix<2, 2, T> transpose_type;
        typedef T               value_type;
        typedef size_t          length_type;

      private:
        col_type value[2];

      public:
        // -- Accesses --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF static constexpr length_type length()
        {
            return 2;
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr col_type       &operator[]( length_type i ) noexcept;
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr col_type const &operator[]( length_type i ) const noexcept;

        // -- Constructors --

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix() = default;

        constexpr matrix( matrix<2, 2, T> const &m );

        constexpr matrix( T scalar );
        constexpr matrix( T const &x1, T const &y1, T const &x2, T const &y2 );
        constexpr matrix( col_type const &v1, col_type const &v2 );

        // -- Conversions --

        template <typename U, typename V, typename M, typename N>
        constexpr matrix( U const &x1, V const &y1, M const &x2, N const &y2 );

        template <typename U, typename V>
        constexpr matrix( vect<2, U> const &v1, vect<2, V> const &v2 );

        // -- Matrix conversions --

        template <typename U>
        constexpr matrix( matrix<2, 2, U> const &m )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<3, 3, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<4, 4, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<2, 3, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<3, 2, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<2, 4, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<4, 2, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<3, 4, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        constexpr matrix( matrix<4, 3, T> const &x )
            : value{ col_type( m[0] ), col_type( m[1] ) }
        {
        }

        // -- Unary arithmetic operators --

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator=( matrix<2, 2, U> const &m )
        {
            this->value[0] = m[0];
            this->value[1] = m[1];

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator+=( U s )
        {
            this->value[0] += scalar;
            this->value[1] += scalar;

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator+=( matrix<2, 2, U> const &m )
        {
            this->value[0] += m[0];
            this->value[1] += m[1];

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator-=( U s )
        {
            this->value[0] -= scalar;
            this->value[1] -= scalar;

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator-=( matrix<2, 2, U> const &m )
        {
            this->value[0] -= m[0];
            this->value[1] -= m[1];

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator*=( U s )
        {
            this->value[0] *= scalar;
            this->value[1] *= scalar;

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator*=( matrix<2, 2, U> const &m )
        {
            return ( *this = *this * m );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator/=( U s )
        {
            this->value[0] /= scalar;
            this->value[1] /= scalar;
            
            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> &operator/=( matrix<2, 2, U> const &m )
        {
            return *this *= inverse( m );
        }
    };

    // -- Unary operators --

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator+( matrix<2, 2, T> const &m )
    {
        return m;
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator-( matrix<2, 2, T> const &m )
    {
        return matrix<2, 2, T>( -m[0], -m[1] );
    }

    // -- Binary operators --

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator+( matrix<2, 2, T> const &m, T scalar )
    {
        return matrix<2, 2, T>( m[0] + scalar, m[1] + scalar );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator+( T scalar, matrix<2, 2, T> const &m )
    {
        return matrix<2, 2, T>( m[0] + scalar, m[1] + scalar );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator+( matrix<2, 2, T> const &m1, matrix<2, 2, T> const &m2 )
    {
        return matrix<2, 2, T>( m1[0] + m2[0], m1[1] + m2[1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator-( matrix<2, 2, T> const &m, T scalar )
    {
        return matrix<2, 2, T>( m[0] - scalar, m[1] - scalar );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator-( T scalar, matrix<2, 2, T> const &m )
    {
        return matrix<2, 2, T>( scalar - m[0], scalar - m[1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator-( matrix<2, 2, T> const &m1, matrix<2, 2, T> const &m2 )
    {
        return matrix<2, 2, T>( m1[0] - m2[0], m1[1] - m2[1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator*( matrix<2, 2, T> const &m, T scalar )
    {
        return matrix<2, 2, T>( m[0] * scalar, m[1] * scalar );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator*( T scalar, matrix<2, 2, T> const &m )
    {
        return matrix<2, 2, T>( m[0] * scalar, m[1] * scalar );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<2, 2, T>::col_type
    operator*( matrix<2, 2, T> const &m, typename matrix<2, 2, T>::row_type const &v )
    {
        return vec<2, T>( m[0][0] * v.x + m[1][0] * v.y, m[0][1] * v.x + m[1][1] * v.y );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<2, 2, T>::row_type
    operator*( typename matrix<2, 2, T>::col_type const &v, matrix<2, 2, T> const &m )
    {
        return vec<2, T>( v.x * m[0][0] + v.y * m[0][1], v.x * m[1][0] + v.y * m[1][1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator*( matrix<2, 2, T> const &m1, matrix<2, 2, T> const &m2 )
    {
        return matrix<2, 2, T>( m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1], m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
                                m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1], m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<3, 2, T> operator*( matrix<2, 2, T> const &m1, matrix<3, 2, T> const &m2 )
    {
        return matrix<3, 2, T>( m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1], m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
                                m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1], m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
                                m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1], m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<4, 2, T> operator*( matrix<2, 2, T> const &m1, matrix<4, 2, T> const &m2 )
    {
        return matrix<4, 2, T>( m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1], m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
                                m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1], m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
                                m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1], m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1],
                                m1[0][0] * m2[3][0] + m1[1][0] * m2[3][1], m1[0][1] * m2[3][0] + m1[1][1] * m2[3][1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator/( matrix<2, 2, T> const &m, T scalar )
    {
        return matrix<2, 2, T>( m[0] / scalar, m[1] / scalar );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator/( T scalar, matrix<2, 2, T> const &m )
    {
        return matrix<2, 2, T>( scalar / m[0], scalar / m[1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<2, 2, T>::col_type
    operator/( matrix<2, 2, T> const &m, typename matrix<2, 2, T>::row_type const &v )
    {
        return inverse( m ) * v;
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<2, 2, T>::row_type
    operator/( typename matrix<2, 2, T>::col_type const &v, matrix<2, 2, T> const &m )
    {
        return v * inverse( m );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<2, 2, T> operator/( matrix<2, 2, T> const &m1, matrix<2, 2, T> const &m2 )
    {
        matrix<2, 2, T> m1_copy( m1 );
        return m1_copy /= m2;
    }

    // -- Boolean operators --

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator==( matrix<2, 2, T> const &m1, matrix<2, 2, T> const &m2 )
    {
        return ( m1[0] == m2[0] ) && ( m1[1] == m2[1] );
    }

    template <typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator!=( matrix<2, 2, T> const &m1, matrix<2, 2, T> const &m2 )
    {
        return ( m1[0] != m2[0] ) || ( m1[1] != m2[1] );
    }
} // namespace glm

#ifndef GLM_EXTERNAL_TEMPLATE
#    include "type_mat2x2.inl"
#endif
