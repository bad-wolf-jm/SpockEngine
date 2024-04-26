#pragma once

#include "core.h"
#include "matrix_4x4.h"

namespace numlua::linalg
{
    using float2x2 = mat<2, 2, float>;

    template <size_t _Rows, size_t _Columns, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr matrix<_Rows, _Columns, T> operator+( matrix<2, 2, T> const &m )
    {
        return m;
    }

    template <size_t _Rows, size_t _Columns, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<_Rows, _Columns, T>::col_type
    operator/( matrix<_Rows, _Columns, T> const &m, typename matrix<_Rows, _Columns, T>::row_type const &v )
    {
        return inverse( m ) * v;
    }

    template <size_t _Rows, size_t _Columns, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr typename matrix<_Rows, _Columns, T>::row_type
    operator/( typename matrix<_Rows, _Columns, T>::col_type const &v, matrix<_Rows, _Columns, T> const &m )
    {
        return v * inverse( m );
    }

    template <size_t _Rows, size_t _Columns, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator==( matrix<_Rows, _Columns, T> const &m1,
                                                                matrix<_Rows, _Columns, T> const &m2 )
    {
        bool result = true;

        for( int i = 0; i < m1.length(); i++ )
            result &= ( m1[i] == m2[i] );

        return result;
    }

    template <size_t _Rows, size_t _Columns, typename T>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator!=( matrix<_Rows, _Columns, T> const &m1,
                                                                matrix<_Rows, _Columns, T> const &m2 )
    {
        bool result = false;

        for( int i = 0; i < m1.length(); i++ )
            result |= ( m1[i] != m2[i] );

        return result;
    }
} // namespace numlua::linalg
