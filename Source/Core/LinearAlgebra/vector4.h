
#pragma once
#include "Core/Cuda/Cuda.h"
#include "core.h"
#include "vector2.h"
#include "vector3.h"
#include <fmt/format.h>

namespace numlua::linalg
{
    template <typename _Ty>
    struct vect<4, _Ty>
    {
        using value_type  = _Ty;
        using length_type = size_t;

        union
        {
            // clang-format off
            struct { _Ty x, y, z, w; };
            struct { _Ty r, g, b, a; };
            struct { _Ty s, t, p, q; };
            // clang-format on
        };

        // clang-format off
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>( _Ty _x, _Ty _y, _Ty _z, _Ty _w )
            : x{ _x } , y{ _y } , z{ _z } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>( _Ty _x )
            : x{ _x } , y{ _x } , z{ _x } , w{ _x } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>(vec2_type<_Ty> const& v, _Ty _z, _Ty _w )
            : x{ v.x } , y{ v.y } , z{ _z } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>(_Ty _x, vec2_type<_Ty> const& v, _Ty _w )
            : x{ _x } , y{ v.x } , z{ v.y } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>(_Ty _x, _Ty _y, vec2_type<_Ty> const& v)
            : x{ _x } , y{ _y } , z{ v.x } , w{ v.y } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>(vec3_type<_Ty> const& v, _Ty _w )
            : x{ v.x } , y{ v.y } , z{ v.z } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>(_Ty _x, vec3_type<_Ty> const& v )
            : x{ _x } , y{ v.x } , z{ v.y } , w{ v.z } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>( vect<4, _Ty> const &v )
            : x{ v.x } , y{ v.y } , z{ v.z } , w{ v.w } { }
        // clang-format on

        template <typename X, typename Y, typename Z, typename W>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty>( X _x, Y _y, Z _z, W _w )
            : x( static_cast<_Ty>( _x ) )
            , y( static_cast<_Ty>( _y ) )
            , z( static_cast<_Ty>( _z ) )
            , w( static_cast<_Ty>( _w ) )
        {
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr _Ty &operator[]( length_type i )
        {
            // GLM_ASSERT_LENGTH( i, this->length() );
            switch( i )
            {
                // clang-format off
            default:
            case 0: return x;
            case 1: return y;
            case 2: return z;
            case 3: return w;
                // clang-format on
            }
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr _Ty &operator[]( length_type i ) const
        {
            // GLM_ASSERT_LENGTH( i, this->length() );
            switch( i )
            {
                // clang-format off
            default:
            case 0: return x;
            case 1: return y;
            case 2: return z;
            case 3: return w;
                // clang-format on
            }
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator=( vect<4, _Ty> const &v )
        {
            this->x = static_cast<_Ty>( v.x );
            this->y = static_cast<_Ty>( v.y );
            this->z = static_cast<_Ty>( v.z );
            this->w = static_cast<_Ty>( v.w );

            return *this;
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator+=( U scalar )
        {
            return ( *this = detail::compute_vec_add<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator+=( vec<4, U, Q> const &v )
        {
            return ( *this = detail::compute_vec_add<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator-=( U scalar )
        {
            return ( *this = detail::compute_vec_sub<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator-=( vec<4, U, Q> const &v )
        {
            return ( *this = detail::compute_vec_sub<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator*=( U scalar )
        {
            return ( *this = detail::compute_vec_mul<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator*=( vec<4, U, Q> const &v )
        {
            return ( *this = detail::compute_vec_mul<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator/=( U scalar )
        {
            return ( *this = detail::compute_vec_div<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator/=( vec<4, U, Q> const &v )
        {
            return ( *this = detail::compute_vec_div<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( v ) ) );
        }

        // -- Increment and decrement operators --

        template <typename T, qualifier Q>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator++()
        {
            ++this->x;
            ++this->y;
            ++this->z;
            ++this->w;
            return *this;
        }

        template <typename T, qualifier Q>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator--()
        {
            --this->x;
            --this->y;
            --this->z;
            --this->w;
            return *this;
        }

        template <typename T, qualifier Q>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator++( int )
        {
            vect<4, _Ty> Result( *this );
            ++*this;
            return Result;
        }

        template <typename T, qualifier Q>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator--( int )
        {
            vect<4, _Ty> Result( *this );
            --*this;
            return Result;
        }

        // -- Unary bit operators --

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator%=( U scalar )
        {
            return ( *this = detail::compute_vec_mod<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator%=( vec<4, U, Q> const &v )
        {
            return ( *this = detail::compute_vec_mod<4, T, Q, detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator&=( U scalar )
        {
            return (
                *this =
                    detail::compute_vec_and<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
                        *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator&=( vec<4, U, Q> const &v )
        {
            return (
                *this =
                    detail::compute_vec_and<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
                        *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator|=( U scalar )
        {
            return (
                *this = detail::compute_vec_or<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
                    *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator|=( vec<4, U, Q> const &v )
        {
            return (
                *this = detail::compute_vec_or<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
                    *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator^=( U scalar )
        {
            return (
                *this =
                    detail::compute_vec_xor<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
                        *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator^=( vec<4, U, Q> const &v )
        {
            return (
                *this =
                    detail::compute_vec_xor<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
                        *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator<<=( U scalar )
        {
            return ( *this = detail::compute_vec_shift_left<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8,
                                                            detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator<<=( vec<4, U, Q> const &v )
        {
            return ( *this = detail::compute_vec_shift_left<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8,
                                                            detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( v ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator>>=( U scalar )
        {
            return ( *this = detail::compute_vec_shift_right<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8,
                                                             detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( scalar ) ) );
        }

        template <typename U>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> &operator>>=( vec<4, U, Q> const &v )
        {
            return ( *this = detail::compute_vec_shift_right<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8,
                                                             detail::is_aligned<Q>::value>::call( *this, vect<4, _Ty>( v ) ) );
        }
    };

    // -- Unary constant operators --
    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator+( vect<4, _Ty> const &v )
    {
        return v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator-( vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( 0 ) -= v;
    }

    // -- Binary arithmetic operators --

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator+( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) += scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator+( vect<4, _Ty> const &v1, vec<1, T, Q> const &v2 )
    {
        return vect<4, _Ty>( v1 ) += v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator+( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( v ) += scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator+( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) += v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator-( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) -= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator-( vect<4, _Ty> const &v1, vec<1, T, Q> const &v2 )
    {
        return vect<4, _Ty>( v1 ) -= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator-( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) -= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator-( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) -= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator*( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) *= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator*( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( v ) *= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator*( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) *= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator/( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) /= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator/( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) /= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator/( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) /= v2;
    }

    // -- Binary bit operators --

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator%( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) %= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator%( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) %= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator%( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) %= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator&( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) &= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator&( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) &= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator&( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) &= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator|( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) |= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator|( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) |= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator|( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) |= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator^( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) ^= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator^( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) ^= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator^( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) ^= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator<<( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) <<= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator<<( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) <<= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator<<( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) <<= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator>>( vect<4, _Ty> const &v, T scalar )
    {
        return vect<4, _Ty>( v ) >>= scalar;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator>>( T scalar, vect<4, _Ty> const &v )
    {
        return vect<4, _Ty>( scalar ) >>= v;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty> operator>>( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return vect<4, _Ty>( v1 ) >>= v2;
    }

    template <typename _Ty>
    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<4, _Ty> operator~( vect<4, _Ty> const &v )
    {
        return detail::compute_vec_bitwise_not<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
            v );
    }

    // -- Boolean operators --

    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator==( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return detail::compute_vec_equal<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call( v1,
                                                                                                                                  v2 );
    }

    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr bool operator!=( vect<4, _Ty> const &v1, vect<4, _Ty> const &v2 )
    {
        return detail::compute_vec_nequal<4, T, Q, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
            v1, v2 );
    }

    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec<4, bool, Q> operator&&( vec<4, bool, Q> const &v1, vec<4, bool, Q> const &v2 )
    {
        return vec<4, bool, Q>( v1.x && v2.x, v1.y && v2.y, v1.z && v2.z, v1.w && v2.w );
    }

    SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec<4, bool, Q> operator||( vec<4, bool, Q> const &v1, vec<4, bool, Q> const &v2 )
    {
        return vec<4, bool, Q>( v1.x || v2.x, v1.y || v2.y, v1.z || v2.z, v1.w || v2.w );
    }
} // namespace numlua::linalg

template <typename _Ty>
struct fmt::formatter<numlua::linalg::vect<4, _Ty><_Ty>> : fmt::formatter<std::string_view>
{
    auto format( numlua::linalg::vect<4, _Ty><_Ty> value, format_context &ctx ) -> format_context::iterator
    {
        auto const &formattedValue = fmt::format( "vec4({}, {}, {}, {})", value.x, value.y, value.z, value.w );

        return fmt::formatter<std::string_view>::format( formattedValue, ctx );
    }
};
