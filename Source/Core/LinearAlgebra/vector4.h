
#pragma once

#include "Core/Cuda/Cuda.h"
#include "vector2.h"
#include "vector3.h"

namespace numlua::linalg
{
    template <typename _Ty>
    struct vec4_type
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
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type( _Ty _x, _Ty _y, _Ty _z, _Ty _w )
            : x{ _x } , y{ _y } , z{ _z } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type( _Ty _x )
            : x{ _x } , y{ _x } , z{ _x } , w{ _x } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type(vec2_type<_Ty> const& v, _Ty _z, _Ty _w )
            : x{ v.x } , y{ v.y } , z{ _z } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type(_Ty _x, vec2_type<_Ty> const& v, _Ty _w )
            : x{ _x } , y{ v.x } , z{ v.y } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type(_Ty _x, _Ty _y, vec2_type<_Ty> const& v)
            : x{ _x } , y{ _y } , z{ v.x } , w{ v.y } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type(vec3_type<_Ty> const& v, _Ty _w )
            : x{ v.x } , y{ v.y } , z{ v.z } , w{ _w } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type(_Ty _x, vec3_type<_Ty> const& v )
            : x{ _x } , y{ v.x } , z{ v.y } , w{ v.z } { }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type( vec4_type const &v )
            : x{ v.x } , y{ v.y } , z{ v.z } , w{ v.w } { }
        // clang-format on

        template <typename X, typename Y, typename Z, typename W>
        SE_CUDA_HOST_DEVICE_FUNCTION_DEF constexpr vec4_type( X _x, Y _y, Z _z, W _w )
            : x( static_cast<_Ty>( _x ) )
            , y( static_cast<_Ty>( _y ) )
            , z( static_cast<_Ty>( _z ) )
            , w( static_cast<_Ty>( _w ) )
        {
        }

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR _Ty &operator[]( length_type i )
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

        SE_CUDA_HOST_DEVICE_FUNCTION_DEF GLM_CONSTEXPR _Ty &operator[]( length_type i ) const
        {
            //GLM_ASSERT_LENGTH( i, this->length() );
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

        GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec4_type &operator=( vec4_type const &v )
        {
            this->x = static_cast<_Ty>( v.x );
            this->y = static_cast<_Ty>( v.y );
            this->z = static_cast<_Ty>( v.z );
            this->w = static_cast<_Ty>( v.w );

            return *this;
        }
    };
} // namespace numlua::linalg
