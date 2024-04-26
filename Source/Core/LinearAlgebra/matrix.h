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

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<M, N, _Ty> transpose( matrix<N, M, _Ty> A )
    {
        matrix<M, N, _Ty> result;

        for( int i = 0; i < N; ++i )
            for( int j = 0; j < M; ++j )
                result[i][j] = A[j][i];

        return result;
    }

    template <typename _Ty>
    LINALG_FUNCTION _Ty determinant( matrix<2, 2, _Ty> m )
    {
        return m[0][0] * m[1][1] - m[1][0] * m[0][1];
    }

    template <typename _Ty>
    LINALG_FUNCTION _Ty determinant( matrix<3, 3, _Ty> m )
    {
        // clang-format off
        return + m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
               - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
               + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
        // clang-format on
    }

    template <typename _Ty>
    LINALG_FUNCTION _Ty determinant( matrix<4, 4, _Ty> m )
    {
        _Ty SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        _Ty SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        _Ty SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        _Ty SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        _Ty SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        _Ty SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

        // clang-format off
        vect<4, _Ty> DetCof(
            + (m[1][1] * SubFactor00 - m[1][2] * SubFactor01 + m[1][3] * SubFactor02),
            - (m[1][0] * SubFactor00 - m[1][2] * SubFactor03 + m[1][3] * SubFactor04),
            + (m[1][0] * SubFactor01 - m[1][1] * SubFactor03 + m[1][3] * SubFactor05),
            - (m[1][0] * SubFactor02 - m[1][1] * SubFactor04 + m[1][2] * SubFactor05));
        // clang-format on

        return m[0][0] * DetCof[0] + m[0][1] * DetCof[1] + m[0][2] * DetCof[2] + m[0][3] * DetCof[3];
    }

    template <typename _Ty>
    LINALG_FUNCTION matrix<2, 2, _Ty> inverse( matrix<2, 2, _Ty> m )
    {
        matrix<2, 2, _Ty> result;

        _Ty OneOverDeterminant = static_cast<_Ty>( 1 ) / ( m[0][0] * m[1][1] - m[1][0] * m[0][1] );

        // clang-format off
        matrix<2, 2, _Ty> result( 
            +m[1][1] * OneOverDeterminant, -m[0][1] * OneOverDeterminant,
            -m[1][0] * OneOverDeterminant, +m[0][0] * OneOverDeterminant );
        // clang-format on

        return result;
    }

    template <typename _Ty>
    LINALG_FUNCTION matrix<3, 3, _Ty> inverse( matrix<3, 3, _Ty> const &m )
    {
        _Ty OneOverDeterminant = static_cast<T>( 1 ) / ( +m[0][0] * ( m[1][1] * m[2][2] - m[2][1] * m[1][2] ) -
                                                         m[1][0] * ( m[0][1] * m[2][2] - m[2][1] * m[0][2] ) +
                                                         m[2][0] * ( m[0][1] * m[1][2] - m[1][1] * m[0][2] ) );

        matrix<3, 3, _Ty> result;
        result[0][0] = +( m[1][1] * m[2][2] - m[2][1] * m[1][2] );
        result[1][0] = -( m[1][0] * m[2][2] - m[2][0] * m[1][2] );
        result[2][0] = +( m[1][0] * m[2][1] - m[2][0] * m[1][1] );
        result[0][1] = -( m[0][1] * m[2][2] - m[2][1] * m[0][2] );
        result[1][1] = +( m[0][0] * m[2][2] - m[2][0] * m[0][2] );
        result[2][1] = -( m[0][0] * m[2][1] - m[2][0] * m[0][1] );
        result[0][2] = +( m[0][1] * m[1][2] - m[1][1] * m[0][2] );
        result[1][2] = -( m[0][0] * m[1][2] - m[1][0] * m[0][2] );
        result[2][2] = +( m[0][0] * m[1][1] - m[1][0] * m[0][1] );

        return result *= OneOverDeterminant;
        ;
    }

    template <typename _Ty>
    LINALG_FUNCTION matrix<4, 4, _Ty> inverse( matrix<4, 4, _Ty> const &m )
    {
        _Ty Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        _Ty Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
        _Ty Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

        _Ty Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        _Ty Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
        _Ty Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

        _Ty Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        _Ty Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
        _Ty Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

        _Ty Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        _Ty Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
        _Ty Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

        _Ty Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        _Ty Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
        _Ty Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

        _Ty Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
        _Ty Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
        _Ty Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

        vect<4, _Ty> Fac0( Coef00, Coef00, Coef02, Coef03 );
        vect<4, _Ty> Fac1( Coef04, Coef04, Coef06, Coef07 );
        vect<4, _Ty> Fac2( Coef08, Coef08, Coef10, Coef11 );
        vect<4, _Ty> Fac3( Coef12, Coef12, Coef14, Coef15 );
        vect<4, _Ty> Fac4( Coef16, Coef16, Coef18, Coef19 );
        vect<4, _Ty> Fac5( Coef20, Coef20, Coef22, Coef23 );

        vect<4, _Ty> Vec0( m[1][0], m[0][0], m[0][0], m[0][0] );
        vect<4, _Ty> Vec1( m[1][1], m[0][1], m[0][1], m[0][1] );
        vect<4, _Ty> Vec2( m[1][2], m[0][2], m[0][2], m[0][2] );
        vect<4, _Ty> Vec3( m[1][3], m[0][3], m[0][3], m[0][3] );

        vect<4, _Ty> Inv0( Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2 );
        vect<4, _Ty> Inv1( Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4 );
        vect<4, _Ty> Inv2( Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5 );
        vect<4, _Ty> Inv3( Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5 );

        vect<4, _Ty>      SignA( +1, -1, +1, -1 );
        vect<4, _Ty>      SignB( -1, +1, -1, +1 );
        matrix<4, 4, _Ty> Inverse( Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB );

        vect<4, _Ty> Row0( Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0] );

        vect<4, _Ty> Dot0( m[0] * Row0 );
        _Ty          Dot1 = ( Dot0.x + Dot0.y ) + ( Dot0.z + Dot0.w );

        _Ty OneOverDeterminant = static_cast<T>( 1 ) / Dot1;

        return Inverse * OneOverDeterminant;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator-( matrix<N, M, _Ty> A, _Ty scalar )
    {
        matrix<N, M, _Ty> result( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] -= scalar;

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator-( _Ty scalar, matrix<N, M, _Ty> A )
    {
        matrix<N, M, _Ty> result;

        for( int i = 0; i < result.length(); ++i )
            result[i] = scalar - A[i];

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator-( matrix<N, M, _Ty> A, matrix<N, M, _Ty> B )
    {
        matrix<N, M, _Ty> result( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] -= B[i];

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator+( matrix<N, M, _Ty> A, _Ty scalar )
    {
        matrix<N, M, _Ty> result( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] += scalar;

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator+( _Ty scalar, matrix<N, M, _Ty> A )
    {
        matrix<N, M, _Ty> result( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] += scalar;

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator+( matrix<N, M, _Ty> A, matrix<N, M, _Ty> B )
    {
        matrix<N, M, _Ty> result( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] += B[i];

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator*( matrix<N, M, _Ty> A, _Ty scalar )
    {
        matrix<N, M, _Ty> result( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] *= scalar;

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION matrix<N, M, _Ty> operator*( _Ty scalar, matrix<N, M, _Ty> A )
    {
        matrix<N, M, _Ty> result( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] *= scalar;

        return result;
    }

    template <size_t N, size_t M, typename _Ty>
    LINALG_FUNCTION vect<M, _Ty> operator*( vect<N, _Ty> v, matrix<N, M, _Ty> A )
    {
        vect<M, _Ty> result;

        for( int i = 0; i < result.length(); ++i )
            result[i] = dot( v, A[i] );

        return result;
    }

    template <size_t N, size_t M, size_t K, typename _Ty>
    LINALG_FUNCTION vect<N, _Ty> operator*( matrix<N, M, _Ty> A, vect<M, _Ty> v )
    {
        vect<N, _Ty>      result;
        matrix<M, N, _Ty> AT = transpose( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] = dot( AT[i], v );

        return result;
    }

    template <size_t N, size_t M, size_t K, typename _Ty>
    LINALG_FUNCTION matrix<N, K, _Ty> operator*( matrix<N, M, _Ty> A, matrix<M, K, _Ty> B )
    {
        matrix<N, K, _Ty> result;
        matrix<M, N, _Ty> AT = transpose( A );

        for( int i = 0; i < result.length(); ++i )
            result[i] = dot( AT[i], B[i] );

        return result;
    }

    template <size_t N, size_t M, typename T>
    LINALG_FUNCTION matrix<N, M, T> operator/( matrix<N, M, T> const &m, T scalar )
    {
       matrix<N, M, _Ty> result( m );

        for( int i = 0; i < result.length(); ++i )
            result[i] /= scalar;

        return result;
    }

    template <size_t N, size_t M, typename T>
    LINALG_FUNCTION matrix<N, M, T> operator/( T scalar, matrix<N, M, T> const &m )
    {
        matrix<N, M, _Ty> result;

        for( int i = 0; i < result.length(); ++i )
            result[i] = scalar / m[i];

        return result;
    }

    template <size_t N, size_t M, typename T>
    LINALG_FUNCTION matrix<N, M,  T> operator/( matrix<N, M, T> const &m1, matrix<N, M, T> const &m2 )
    {
        matrix<2, 2, T> m1_copy( m1 );

        return m1_copy /= m2;
    }

} // namespace numlua::linalg
