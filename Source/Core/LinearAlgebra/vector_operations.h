
#pragma once

#include "Core/Cuda/Cuda.h"
#include "core.h"
#include <functional>

namespace numlua::linalg
{
    namespace detail
    {
        template <length_t L, typename T, bool UseSimd>
        struct compute_vec_add
        {
        };

        template <length_t L, typename T, bool UseSimd>
        struct compute_vec_sub
        {
        };

        template <length_t L, typename T, bool UseSimd>
        struct compute_vec_mul
        {
        };

        template <length_t L, typename T, bool UseSimd>
        struct compute_vec_div
        {
        };

        template <length_t L, typename T, bool UseSimd>
        struct compute_vec_mod
        {
        };

        template <length_t L, typename T, bool UseSimd>
        struct compute_splat
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_and
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_or
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_xor
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_shift_left
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_shift_right
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_equal
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_nequal
        {
        };

        template <length_t L, typename T, int IsInt, std::size_t Size, bool UseSimd>
        struct compute_vec_bitwise_not
        {
        };

        template <length_t L, typename T>
        struct compute_vec_add<L, T, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vec, L, T, Q>::call( std::plus<T>(), a, b );
            }
        };

        template <length_t L, typename T>
        struct compute_vec_sub<L, T, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vec, L, T, Q>::call( std::minus<T>(), a, b );
            }
        };

        template <length_t L, typename T>
        struct compute_vec_mul<L, T, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vec, L, T, Q>::call( std::multiplies<T>(), a, b );
            }
        };

        template <length_t L, typename T>
        struct compute_vec_div<L, T, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vec, L, T, Q>::call( std::divides<T>(), a, b );
            }
        };

        template <length_t L, typename T>
        struct compute_vec_mod<L, T, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vec, L, T, Q>::call( std::modulus<T>(), a, b );
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_and<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                vect<L, T> v( a );
                for( length_t i = 0; i < L; ++i )
                    v[i] &= static_cast<T>( b[i] );
                return v;
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_or<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                vect<L, T> v( a );
                for( length_t i = 0; i < L; ++i )
                    v[i] |= static_cast<T>( b[i] );
                return v;
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_xor<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                vect<L, T> v( a );
                for( length_t i = 0; i < L; ++i )
                    v[i] ^= static_cast<T>( b[i] );
                return v;
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_shift_left<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                vect<L, T> v( a );
                for( length_t i = 0; i < L; ++i )
                    v[i] <<= static_cast<T>( b[i] );
                return v;
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_shift_right<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                vect<L, T> v( a );
                for( length_t i = 0; i < L; ++i )
                    v[i] >>= static_cast<T>( b[i] );
                return v;
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_equal<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static bool call( vect<L, T> const &v1, vect<L, T> const &v2 )
            {
                bool b = true;
                for( length_t i = 0; b && i < L; ++i )
                    b = (v1[i] == v2[i]);
                return b;
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_nequal<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static bool call( vec<4, T, Q> const &v1, vec<4, T, Q> const &v2 )
            {
                return !compute_vec_equal<L, T, detail::is_int<T>::value, sizeof( T ) * 8, detail::is_aligned<Q>::value>::call(
                    v1, v2 );
            }
        };

        template <length_t L, typename T, int IsInt, std::size_t Size>
        struct compute_vec_bitwise_not<L, T, IsInt, Size, false>
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a )
            {
                vect<L, T> v( a );
                for( length_t i = 0; i < L; ++i )
                    v[i] = ~v[i];
                return v;
            }
        };

    } // namespace detail
} // namespace numlua::linalg
