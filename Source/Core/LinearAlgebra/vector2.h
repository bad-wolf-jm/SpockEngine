#pragma once

#include "core.h"

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
    };

} // namespace numlua::linalg
