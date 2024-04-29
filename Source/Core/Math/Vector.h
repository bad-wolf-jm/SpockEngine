/// @file   Vector.h
///
/// @brief  Wrapper interface for glm vector types
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include "glm.h"

/**
 * \namespace math
 *
 * @brief Collection of math related functions
 *
 * Long doc of Records.
 */
namespace math
{

    // /** @brief 2 dimensional vector with integer coordinates. Compatible with GLSL type `ivec2`
    //  * and HLSL type int2.
    //  */
    using ivec2 = glm::ivec2;

    // /** @brief 2 dimensional vector with unsigned integer coordinates. Compatible with GLSL type `uvec2`
    //  * and HLSL type uint2.
    //  */
    // using uvec2 = glm::uvec2;

    /** @brief 2 dimensional vector with floating point coordinates. Compatible with GLSL type `vec2`
     * and HLSL type float2.
     */
    using vec2 = glm::vec2;

    // 3 dimensional vectors

    // /** @brief 3 dimensional vector with integer coordinates. Compatible with GLSL type `ivec3`
    //  * and HLSL type int3.
    //  */
    using ivec3 = glm::ivec3;

    // /** @brief 3 dimensional vector with unsigned integer coordinates. Compatible with GLSL type `uvec3`
    //  * and HLSL type uint3.
    //  */
    // using uvec3 = glm::uvec3;

    /** @brief 3 dimensional vector with floating point coordinates. Compatible with GLSL type `vec3`
     * and HLSL type float3.
     */
    using vec3 = glm::vec3;

    // 4 dimensional vectors

    // /** @brief 4 dimensional vector with integer coordinates. Compatible with GLSL type `ivec4`
    //  * and HLSL type int4.
    //  */
    using ivec4 = glm::ivec4;

    // /** @brief 4 dimensional vector with unsigned integer coordinates. Compatible with GLSL type `uvec4`
    //  * and HLSL type uint4.
    //  */
    // using uvec4 = glm::uvec4;

    /** @brief 5 dimensional vector with floating point coordinates. Compatible with GLSL type `vec4`
     * and HLSL type float4.
     */
    using vec4 = glm::vec4;

    // Quaternion types

    /** @brief Quaternion type with floating point coordinates. */
    using quat = glm::quat;

    /** @brief Dual quaternion type with floating point coordinates. */
    using dualquat = glm::dualquat;

    // 3 dimensional coordinate axes
    /** @brief Standard basis x axis.
     *
     * Equal to `vec3(1.0f, 0.0f, 0.0f)`.
     */
    inline vec3 x_axis()
    {
        return vec3( 1.0f, 0.0f, 0.0f );
    }

    /** @brief Standard basis y axis
     *
     * Equal to `vec3(0.0f, 1.0f, 0.0f)`.
     */
    inline vec3 y_axis()
    {
        return vec3( 0.0f, 1.0f, 0.0f );
    }

    /** @brief Standard basis z axis.
     *
     * Equal to `vec3(0.0f, 0.0f, 1.0f)`.
     */
    inline vec3 z_axis()
    {
        return vec3( 0.0f, 0.0f, 1.0f );
    }

    /** @brief Euclidean length of input vector. */
    template <typename T>
    inline float length( const T &vector )
    {
        return glm::length( vector );
    }

    /** @brief Euclidean length of input vector. */
    template <typename T>
    inline float length2( const T &vector )
    {
        return glm::length2( vector );
    }

    /** @brief Euclidean length of input vector. */
    template <typename T>
    inline float dist2( const T &vector0, const T &vector1 )
    {
        return glm::distance2( vector0, vector1 );
    }

    /** @brief Euclidean length of input vector. */
    inline float det( const vec2 &vector0, const vec2 &vector1 )
    {
        return ( vector0.x * vector1.y ) - ( vector0.y * vector1.x );
    }

    /** @brief Unit vector in the direction of V. */
    template <typename T>
    T normalize( const T &vector )
    {
        return glm::normalize( vector );
    }

    /** @brief Standard dot-product of v1 and v2. */
    template <typename T>
    inline float dot( const T &vector1, const T &vector2 )
    {
        return glm::dot( vector1, vector2 );
    }

    /** @brief Standard cross-product of v1 and v2. */
    inline vec3 cross( const vec3 &vector1, const vec3 &vector2 )
    {
        return glm::cross( vector1, vector2 );
    }

    /** @brief Returns a vector normal to v1. */
    inline vec2 perpendicular( const vec2 &vector1 )
    {
        return { -vector1.y, vector1.x };
    }

    /** @brief Color conversion. */
    inline vec3 hsv_to_rgb( float hue, float saturation, float value )
    {
        hue -= glm::floor( hue / 360.0 ) * 360.0f;

        if( hue < 0 )
            hue += 360.0f;

        int h = static_cast<int>( hue / 60.0f ) % 6;

        float f = hue / 60.0f - h;
        float p = value * ( 1.0f - saturation );
        float q = value * ( 1.0f - f * saturation );
        float t = value * ( 1.0f - ( 1.0f - f ) * saturation );

        switch( h )
        {
        case 0:
            return vec3( value, t, p );
        case 1:
            return vec3( q, value, p );
        case 2:
            return vec3( p, value, t );
        case 3:
            return vec3( p, q, value );
        case 4:
            return vec3( t, p, value );
        case 5:
            return vec3( value, p, q );
        default:
            return vec3( 0, 0, 0 );
        }
    }

    /** @brief Interpolates the vectors `vector1` and `vector2` using ratio `coefficient` */
    template <typename T>
    T mix( T vector1, T vector2, float coefficient )
    {
        return glm::mix( vector1, vector2, coefficient );
    }

    /** @brief Sperically interpolates the quaternions `quaternion1` and `quaternion2` using ratio `coefficient` */
    template <typename T>
    T slerp( T quaternion1, T quaternion2, float coefficient )
    {
        return glm::slerp( quaternion1, quaternion2, coefficient );
    }

} // namespace math
