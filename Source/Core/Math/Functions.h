#pragma once

#include "Vector.h"

namespace math
{

    template <typename _Ty> inline _Ty mod( _Ty x, _Ty y ) { return glm::mod( x, y ); }

    inline float log( float x ) { return glm::log( x ); }

    inline float abs( float x ) { return glm::abs( x ); }

    inline float sin( float x ) { return glm::sin( x ); }

    inline float cos( float x ) { return glm::cos( x ); }

    template <typename T> T max( T x, T y ) { return glm::max( x, y ); }

    template <typename T> T min( T x, T y ) { return glm::min( x, y ); }

    template <typename T> float *ptr( T &V ) { return glm::value_ptr( V ); }

    template <typename T> T radians( T V ) { return glm::radians( V ); }

    template <typename T> T degrees( T V ) { return glm::degrees( V ); }

    namespace literals
    {
        inline float operator"" _degf( long double v ) { return glm::radians( (float)v ); }

        inline vec3 operator"" _rgbf( unsigned long long value )
        {
            return { ( uint8_t( value >> 16 ) / 255.0f ), ( uint8_t( value >> 8 ) / 255.0f ), ( uint8_t( value ) / 255.0f ) };
        }

        inline vec4 operator"" _rgbaf( unsigned long long value )
        {
            return { ( uint8_t( value >> 24 ) / 255.0f ), ( uint8_t( value >> 16 ) / 255.0f ), ( uint8_t( value >> 8 ) / 255.0f ), ( uint8_t( value ) / 255.0f ) };
        }
    } // namespace literals
} // namespace math
