
#pragma once
#include "core.h"
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
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty>( _Ty _x, _Ty _y, _Ty _z )
            : x{ _x } , y{ _y } , z{ _z } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty>( _Ty _x )
            : x{ _x } , y{ _x } , z{ _x } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty>(vect<2, _Ty> const& v, _Ty _z )
            : x{ v.x } , y{ v.y } , z{ _z } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty>(_Ty _x, vect<2,_Ty> const& v )
            : x{ _x } , y{ v.x } , z{ v.y } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty>( vect<3, _Ty> const &v )
            : x{ v.x } , y{ v.y } , z{ v.z } { }
        // clang-format on

        template <typename X, typename Y, typename Z, typename W>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty>( X _x, Y _y, Z _z )
            : x( static_cast<_Ty>( _x ) )
            , y( static_cast<_Ty>( _y ) )
            , z( static_cast<_Ty>( _z ) )
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
            case 2: return z;
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
            case 2: return z;
                // clang-format on
            }
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vect<3, _Ty> &operator=( vect<3, _Ty> const &v )
        {
            this->x = static_cast<_Ty>( v.x );
            this->y = static_cast<_Ty>( v.y );
            this->z = static_cast<_Ty>( v.z );

            return *this;
        }

    };
} // namespace numlua::linalg
