
#pragma once
#include "core.h"
#include "vector_operations.h"

namespace numlua::linalg
{
    template <typename _Ty>
    struct vect<3, _Ty>
    {
        using value_type = _Ty;

        union
        {
            // clang-format off
            struct { _Ty x, y, z; };
            struct { _Ty r, g, b; };
            struct { _Ty s, t, p; };
            // clang-format on
        };

        // clang-format off
        LINALG_FUNCTION vect<3, _Ty>( _Ty _x, _Ty _y, _Ty _z )
            : x{ _x } , y{ _y } , z{ _z } { }

        LINALG_FUNCTION vect<3, _Ty>( _Ty _x )
            : x{ _x } , y{ _x } , z{ _x } { }

        LINALG_FUNCTION vect<3, _Ty>( vect<4, _Ty> const &v )
            : x{ v.x } , y{ v.y } , z{ v.z } { }

        LINALG_FUNCTION vect<3, _Ty>(vect<2, _Ty> const& v, _Ty _z )
            : x{ v.x } , y{ v.y } , z{ _z } { }

        LINALG_FUNCTION vect<3, _Ty>(_Ty _x, vect<2,_Ty> const& v )
            : x{ _x } , y{ v.x } , z{ v.y } { }

        LINALG_FUNCTION vect<3, _Ty>( vect<3, _Ty> const &v )
            : x{ v.x } , y{ v.y } , z{ v.z } { }
        // clang-format on

        template <typename X, typename Y, typename Z, typename W>
        LINALG_FUNCTION vect<3, _Ty>( X _x, Y _y, Z _z )
            : x( static_cast<_Ty>( _x ) )
            , y( static_cast<_Ty>( _y ) )
            , z( static_cast<_Ty>( _z ) )
        {
        }

        LINALG_FUNCTION _Ty &operator[]( size_t i )
        {
            // GLM_ASSERT_LENGTH( i, this->length() );
            switch( i )
            {
                // clang-format off
            default:
            case 0: return x;
            case 1: return y;
            case 2: return z;
                // clang-format on
            }
        }

        LINALG_FUNCTION _Ty const &operator[]( size_t i ) const
        {
            // GLM_ASSERT_LENGTH( i, this->length() );
            switch( i )
            {
                // clang-format off
            default:
            case 0: return x;
            case 1: return y;
            case 2: return z;
                // clang-format on
            }
        }

        LINALG_FUNCTION vect<3, _Ty> &operator=( vect<3, _Ty> const &v )
        {
            this->x = static_cast<_Ty>( v.x );
            this->y = static_cast<_Ty>( v.y );
            this->z = static_cast<_Ty>( v.z );

            return *this;
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator+=( U scalar )
        {
            return ( *this = detail::compute_vec_add<3, _Ty>::call( *this, vect<3, _Ty>( scalar ) ) );
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator+=( vect<3, U> const &v )
        {
            return ( *this = detail::compute_vec_add<3, _Ty>::call( *this, vect<3, _Ty>( v ) ) );
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator-=( U scalar )
        {
            return ( *this = detail::compute_vec_sub<3, _Ty>::call( *this, vect<3, _Ty>( scalar ) ) );
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator-=( vect<3, U> const &v )
        {
            return ( *this = detail::compute_vec_sub<3, _Ty>::call( *this, vect<3, _Ty>( v ) ) );
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator*=( U scalar )
        {
            return ( *this = detail::compute_vec_mul<3, _Ty>::call( *this, vect<3, _Ty>( scalar ) ) );
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator*=( vect<3, U> const &v )
        {
            return ( *this = detail::compute_vec_mul<3, _Ty>::call( *this, vect<3, _Ty>( v ) ) );
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator/=( U scalar )
        {
            return ( *this = detail::compute_vec_div<3, _Ty>::call( *this, vect<3, _Ty>( scalar ) ) );
        }

        template <typename U>
        LINALG_FUNCTION vect<3, _Ty> &operator/=( vect<3, U> const &v )
        {
            return ( *this = detail::compute_vec_div<3, _Ty>::call( *this, vect<3, _Ty>( v ) ) );
        }
    };

    namespace detail
    {
        template <template <length_t L, typename T> class vec, typename R, typename T>
        struct functor1<vec, 3, R, T>
        {
            LINALG_FUNCTION static vec<3, R> call( R ( *Func )( T x ), vec<3, T> const &v )
            {
                return vec<3, R>( Func( v.x ), Func( v.y ), Func( v.z ) );
            }
        };

        template <template <length_t L, typename T> class vec, typename T>
        struct functor2<vec, 3, T>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vec<3, T> call( T ( *Func )( T x, T y ), vec<3, T> const &a, vec<3, T> const &b )
            {
                return vec<3, T>( Func( a.x, b.x ), Func( a.y, b.y ), Func( a.z, b.z ) );
            }

            template <class Fct>
            LINALG_FUNCTION static vec<3, T> call( Fct Func, vec<3, T> const &a, vec<3, T> const &b )
            {
                return vec<3, T>( Func( a.x, b.x ), Func( a.y, b.y ), Func( a.z, b.z ) );
            }
        };
    } // namespace detail
} // namespace numlua::linalg
