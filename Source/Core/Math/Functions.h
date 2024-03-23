#pragma once

#include "Matrix.h"
#include "Vector.h"

namespace math
{
    template <typename _Ty>
    inline _Ty mod( _Ty x, _Ty y )
    {
        return glm::mod( x, y );
    }

    inline float log( float x )
    {
        return glm::log( x );
    }

    inline float abs( float x )
    {
        return glm::abs( x );
    }

    inline float sin( float x )
    {
        return glm::sin( x );
    }

    inline float cos( float x )
    {
        return glm::cos( x );
    }

    template <typename T>
    T max( T x, T y )
    {
        return glm::max( x, y );
    }

    template <typename T>
    T min( T x, T y )
    {
        return glm::min( x, y );
    }

    template <typename T>
    float *ptr( T &V )
    {
        return glm::value_ptr( V );
    }
    template <typename T>
    float *ptr( T const &V )
    {
        return glm::value_ptr( V );
    }

    template <typename T>
    vec3 make_vec3( T *V )
    {
        return vec3( glm::make_vec3( V ) );
    }
    
    template <typename T>
    vec4 make_vec4( T *V )
    {
        return vec4( glm::make_vec4( V ) );
    }

    template <typename T>
    mat4 make_mat3x3( T *V )
    {
        return mat4( glm::make_mat3x3( V ) );
    }

    template <typename T>
    mat4 make_mat4x4( T *V )
    {
        return mat4( glm::make_mat4x4( V ) );
    }

    template <typename T>
    quat make_quat( T *V )
    {
        return quat( glm::make_quat( V ) );
    }

    template <typename T>
    T radians( T V )
    {
        return glm::radians( V );
    }

    template <typename T>
    T degrees( T V )
    {
        return glm::degrees( V );
    }
} // namespace math
