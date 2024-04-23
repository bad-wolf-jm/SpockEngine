
#pragma once

#include "core.h"
#include <functional>

namespace numlua::linalg
{
    namespace detail
    {
        // clang-format off
        // template <size_t L, typename T> struct compute_vec_add { };
        // template <size_t L, typename T> struct compute_vec_sub { };
        // template <size_t L, typename T> struct compute_vec_mul { };
        // template <size_t L, typename T> struct compute_vec_div { };
        // template <size_t L, typename T> struct compute_vec_mod { };
        // template <size_t L, typename T> struct compute_splat { };
        // template <size_t L, typename T, int IsInt, std::size_t Size, bool UseSimd> struct compute_vec_and { };
        // template <size_t L, typename T, int IsInt, std::size_t Size, bool UseSimd> struct compute_vec_or { };
        // template <size_t L, typename T, int IsInt, std::size_t Size, bool UseSimd> struct compute_vec_xor { };
        // template <size_t L, typename T, int IsInt, std::size_t Size, bool UseSimd> struct compute_vec_shift_left { };
        // template <size_t L, typename T, int IsInt, std::size_t Size, bool UseSimd> struct compute_vec_shift_right { };
        // template <size_t L, typename T> struct compute_vec_equal { };
        // template <size_t L, typename T> struct compute_vec_nequal { };
        // template <size_t L, typename T, int IsInt, std::size_t Size, bool UseSimd> struct compute_vec_bitwise_not { };
        // clang-format on

        template <size_t L, typename T>
        struct compute_vec_add
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vect, L, T>::call( std::plus<T>(), a, b );
            }
        };

        template <size_t L, typename T>
        struct compute_vec_sub
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vect, L, T>::call( std::minus<T>(), a, b );
            }
        };

        template <size_t L, typename T>
        struct compute_vec_mul
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vect, L, T>::call( std::multiplies<T>(), a, b );
            }
        };

        template <size_t L, typename T>
        struct compute_vec_div
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
            {
                return detail::functor2<vect, L, T>::call( std::divides<T>(), a, b );
            }
        };

        // template <size_t L, typename T>
        // struct compute_vec_mod<L, T, false>
        // {
        //     SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
        //     {
        //         return detail::functor2<vect, L, T>::call( std::modulus<T>(), a, b );
        //     }
        // };

        // template <size_t L, typename T, int IsInt, std::size_t Size>
        // struct compute_vec_and<L, T, IsInt, Size, false>
        // {
        //     SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
        //     {
        //         vect<L, T> v( a );
        //         for( size_t i = 0; i < L; ++i )
        //             v[i] &= static_cast<T>( b[i] );
        //         return v;
        //     }
        // };

        // template <size_t L, typename T, int IsInt, std::size_t Size>
        // struct compute_vec_or<L, T, IsInt, Size, false>
        // {
        //     SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
        //     {
        //         vect<L, T> v( a );
        //         for( size_t i = 0; i < L; ++i )
        //             v[i] |= static_cast<T>( b[i] );
        //         return v;
        //     }
        // };

        // template <size_t L, typename T, int IsInt, std::size_t Size>
        // struct compute_vec_xor<L, T, IsInt, Size, false>
        // {
        //     SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
        //     {
        //         vect<L, T> v( a );
        //         for( size_t i = 0; i < L; ++i )
        //             v[i] ^= static_cast<T>( b[i] );
        //         return v;
        //     }
        // };

        // template <size_t L, typename T, int IsInt, std::size_t Size>
        // struct compute_vec_shift_left<L, T, IsInt, Size, false>
        // {
        //     SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
        //     {
        //         vect<L, T> v( a );
        //         for( size_t i = 0; i < L; ++i )
        //             v[i] <<= static_cast<T>( b[i] );
        //         return v;
        //     }
        // };

        // template <size_t L, typename T, int IsInt, std::size_t Size>
        // struct compute_vec_shift_right<L, T, IsInt, Size, false>
        // {
        //     SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a, vect<L, T> const &b )
        //     {
        //         vect<L, T> v( a );
        //         for( size_t i = 0; i < L; ++i )
        //             v[i] >>= static_cast<T>( b[i] );
        //         return v;
        //     }
        // };

        template <size_t L, typename T>
        struct compute_vec_equal
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static bool call( vect<L, T> const &v1, vect<L, T> const &v2 )
            {
                bool b = true;
                for( size_t i = 0; b && i < L; ++i )
                    b = ( v1[i] == v2[i] );
                
                return b;
            }
        };

        template <size_t L, typename T>
        struct compute_vec_nequal
        {
            SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static bool call( vect<4, T> const &v1, vect<4, T> const &v2 )
            {
                return !compute_vec_equal<L, T>::call( v1, v2 );
            }
        };

        // template <size_t L, typename T, int IsInt, std::size_t Size>
        // struct compute_vec_bitwise_not<L, T, IsInt, Size, false>
        // {
        //     SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr static vect<L, T> call( vect<L, T> const &a )
        //     {
        //         vect<L, T> v( a );
        //         for( size_t i = 0; i < L; ++i )
        //             v[i] = ~v[i];
        //         return v;
        //     }
        // };

    } // namespace detail
} // namespace numlua::linalg
