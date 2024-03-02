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

    /** @brief 2 dimensional vector with integer coordinates. Compatible with GLSL type `ivec2`
     * and HLSL type int2.
     */
    using ivec2 = glm::ivec2;

    /** @brief 2 dimensional vector with unsigned integer coordinates. Compatible with GLSL type `uvec2`
     * and HLSL type uint2.
     */
    using uvec2 = glm::uvec2;

    /** @brief 2 dimensional vector with floating point coordinates. Compatible with GLSL type `vec2`
     * and HLSL type float2.
     */
    using vec2 = glm::vec2;

    // 3 dimensional vectors

    /** @brief 3 dimensional vector with integer coordinates. Compatible with GLSL type `ivec3`
     * and HLSL type int3.
     */
    using ivec3 = glm::ivec3;

    /** @brief 3 dimensional vector with unsigned integer coordinates. Compatible with GLSL type `uvec3`
     * and HLSL type uint3.
     */
    using uvec3 = glm::uvec3;

    /** @brief 3 dimensional vector with floating point coordinates. Compatible with GLSL type `vec3`
     * and HLSL type float3.
     */
    using vec3 = glm::vec3;

    // 4 dimensional vectors

    /** @brief 4 dimensional vector with integer coordinates. Compatible with GLSL type `ivec4`
     * and HLSL type int4.
     */
    using ivec4 = glm::ivec4;

    /** @brief 4 dimensional vector with unsigned integer coordinates. Compatible with GLSL type `uvec4`
     * and HLSL type uint4.
     */
    using uvec4 = glm::uvec4;

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
    inline vec3 x_axis() { return vec3( 1.0f, 0.0f, 0.0f ); }

    /** @brief Standard basis y axis
     *
     * Equal to `vec3(0.0f, 1.0f, 0.0f)`.
     */
    inline vec3 y_axis() { return vec3( 0.0f, 1.0f, 0.0f ); }

    /** @brief Standard basis z axis.
     *
     * Equal to `vec3(0.0f, 0.0f, 1.0f)`.
     */
    inline vec3 z_axis() { return vec3( 0.0f, 0.0f, 1.0f ); }

    /** @brief Euclidean length of input vector. */
    template <typename T>
    inline float length( const T &aVector )
    {
        return glm::length( aVector );
    }

    /** @brief Euclidean length of input vector. */
    template <typename T>
    inline float length2( const T &aVector )
    {
        return glm::length2( aVector );
    }

    /** @brief Euclidean length of input vector. */
    template <typename T>
    inline float dist2( const T &aVector0, const T &aVector1 )
    {
        return glm::distance2( aVector0, aVector1 );
    }

    /** @brief Euclidean length of input vector. */
    inline float det( const vec2 &aVector0, const vec2 &aVector1 )
    {
        return ( aVector0.x * aVector1.y ) - ( aVector0.y * aVector1.x );
    }

    /** @brief Unit vector in the direction of V. */
    template <typename T>
    T normalize( const T &aVector )
    {
        return glm::normalize( aVector );
    }

    /** @brief Standard dot-product of v1 and v2. */
    template <typename T>
    inline float dot( const T &aVector1, const T &aVector2 )
    {
        return glm::dot( aVector1, aVector2 );
    }

    /** @brief Standard cross-product of v1 and v2. */
    inline vec3 cross( const vec3 &aVector1, const vec3 &aVector2 ) { return glm::cross( aVector1, aVector2 ); }

    /** @brief Returns a vector normal to v1. */
    inline vec2 perpendicular( const vec2 &aVector1 ) { return { -aVector1.y, aVector1.x }; }

    /** @brief Color conversion. */
    inline vec3 hsv_to_rgb( float aHue, float aSaturation, float aValue )
    {
        aHue -= glm::floor( aHue / 360.0 ) * 360.0f;

        if( aHue < 0 ) aHue += 360.0f;

        int lH = static_cast<int>( aHue / 60.0f ) % 6;

        float lF = aHue / 60.0f - lH;
        float lP = aValue * ( 1.0f - aSaturation );
        float lQ = aValue * ( 1.0f - lF * aSaturation );
        float lT = aValue * ( 1.0f - ( 1.0f - lF ) * aSaturation );

        switch( lH )
        {
        case 0: return vec3( aValue, lT, lP );
        case 1: return vec3( lQ, aValue, lP );
        case 2: return vec3( lP, aValue, lT );
        case 3: return vec3( lP, lQ, aValue );
        case 4: return vec3( lT, lP, aValue );
        case 5: return vec3( aValue, lP, lQ );
        default: return vec3( 0, 0, 0 );
        }
    }

    /** @brief Interpolates the vectors `aVector1` and `aVector2` using ratio `aCoefficient` */
    template <typename T>
    T mix( T aVector1, T aVector2, float aCoefficient )
    {
        return glm::mix( aVector1, aVector2, aCoefficient );
    }

    /** @brief Sperically interpolates the quaternions `aQuaternion1` and `aQuaternion2` using ratio `aCoefficient` */
    template <typename T>
    T slerp( T aQuaternion1, T aQuaternion2, float aCoefficient )
    {
        return glm::slerp( aQuaternion1, aQuaternion2, aCoefficient );
    }

} // namespace math
