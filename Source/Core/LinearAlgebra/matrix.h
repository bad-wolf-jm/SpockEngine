#pragma once

#include "core.h"
#include "matrix2x2.h"

namespace numlua::linalg
{
    using float2x2 = matrix<2, 2, float>;

    template <size_t _R, size_t _C, typename T>
    LINALG_FUNCTION matrix<_R, _C, T> operator+( matrix<2, 2, T> const &m )
    {
        return m;
    }

    template <size_t _R, size_t _C, typename T>
    LINALG_FUNCTION typename matrix<_R, _C, T>::col_type operator/( matrix<_R, _C, T> const                    &m,
                                                                    typename matrix<_R, _C, T>::row_type const &v )
    {
        return inverse( m ) * v;
    }

    template <size_t _R, size_t _C, typename T>
    LINALG_FUNCTION typename matrix<_R, _C, T>::row_type operator/( typename matrix<_R, _C, T>::col_type const &v,
                                                                    matrix<_R, _C, T> const                    &m )
    {
        return v * inverse( m );
    }

    template <size_t _R, size_t _C, typename T>
    LINALG_FUNCTION bool operator==( matrix<_R, _C, T> const &m1, matrix<_R, _C, T> const &m2 )
    {
        bool result = true;

        for( int i = 0; i < m1.length(); i++ )
            result &= ( m1[i] == m2[i] );

        return result;
    }

    template <size_t _R, size_t _C, typename T>
    LINALG_FUNCTION bool operator!=( matrix<_R, _C, T> const &m1, matrix<_R, _C, T> const &m2 )
    {
        bool result = false;

        for( int i = 0; i < m1.length(); i++ )
            result |= ( m1[i] != m2[i] );

        return result;
    }
} // namespace numlua::linalg
