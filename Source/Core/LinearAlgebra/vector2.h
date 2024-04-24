#pragma once

#include "core.h"
#include "vector_operations.h"

namespace numlua::linalg
{
    template <typename _Ty>
    struct vect<2, _Ty>
    {
        using value_type = _Ty;

        union
        {
            // clang-format off
            struct { _Ty x, y; };
            struct { _Ty r, g; };
            struct { _Ty s, t; };
            // clang-format on
        };
        // clang-format off
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty>( _Ty _x, _Ty _y )
            : x{ _x } , y{ _y } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty>( _Ty _x )
            : x{ _x } , y{ _x } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty>( vect<2, _Ty> const &v )
            : x{ v.x } , y{ v.y } { }
        // clang-format on

        template <typename X, typename Y, typename Z, typename W>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty>( X _x, Y _y )
            : x( static_cast<_Ty>( _x ) )
            , y( static_cast<_Ty>( _y ) )
        {
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr _Ty &operator[]( size_t i )
        {
            // GLM_ASSERT_LENGTH( i, this->length() );
            switch( i )
            {
                // clang-format off
            default:
            case 0: return x;
            case 1: return y;
                // clang-format on
            }
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr _Ty const &operator[]( size_t i ) const
        {
            // GLM_ASSERT_LENGTH( i, this->length() );
            switch( i )
            {
                // clang-format off
            default:
            case 0: return x;
            case 1: return y;
                // clang-format on
            }
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator=( vect<2, _Ty> const &v )
        {
            this->x = static_cast<_Ty>( v.x );
            this->y = static_cast<_Ty>( v.y );

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator+=( U scalar )
        {
            return ( *this = detail::compute_vec_add<2, _Ty>::call( *this, vect<2, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator+=( vect<2, U> const &v )
        {
            return ( *this = detail::compute_vec_add<2, _Ty>::call( *this, vect<2, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator-=( U scalar )
        {
            return ( *this = detail::compute_vec_sub<2, _Ty>::call( *this, vect<2, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator-=( vect<2, U> const &v )
        {
            return ( *this = detail::compute_vec_sub<2, _Ty>::call( *this, vect<2, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator*=( U scalar )
        {
            return ( *this = detail::compute_vec_mul<2, _Ty>::call( *this, vect<2, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator*=( vect<2, U> const &v )
        {
            return ( *this = detail::compute_vec_mul<2, _Ty>::call( *this, vect<2, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator/=( U scalar )
        {
            return ( *this = detail::compute_vec_div<2, _Ty>::call( *this, vect<2, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<2, _Ty> &operator/=( vect<2, U> const &v )
        {
            return ( *this = detail::compute_vec_div<2, _Ty>::call( *this, vect<2, _Ty>( v ) ) );
        }
    };

    namespace detail
    {
        template <template <length_t L, typename T> class vec, typename R, typename T>
        struct functor1<vec, 2, R, T>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vec<2, R> call( R ( *Func )( T x ), vec<2, T> const &v )
            {
                return vec<2, R>( Func( v.x ), Func( v.y ) );
            }
        };

        template <template <length_t L, typename T> class vec, typename T>
        struct functor2<vec, 2, T>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF static vec<2, T> call( T ( *Func )( T x, T y ), vec<2, T> const &a, vec<3, T> const &b )
            {
                return vec<2, T>( Func( a.x, b.x ), Func( a.y, b.y ) );
            }

            template <class Fct>
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vec<2, T> call( Fct Func, vec<2, T> const &a, vec<2, T> const &b )
            {
                return vec<2, T>( Func( a.x, b.x ), Func( a.y, b.y ) );
            }
        };
    } // namespace detail
} // namespace numlua::linalg
