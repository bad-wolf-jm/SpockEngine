using System;

namespace SpockEngine.Math
{
    // /** @brief Euclidean length of input vector. */
    // template <typename T>
    // inline float length( const T &aVector )
    // {
    //     return glm::length( aVector );
    // }

    // /** @brief Euclidean length of input vector. */
    // template <typename T>
    // inline float length2( const T &aVector )
    // {
    //     return glm::length2( aVector );
    // }

    // /** @brief Euclidean length of input vector. */
    // template <typename T>
    // inline float dist2( const T &aVector0, const T &aVector1 )
    // {
    //     return glm::distance2( aVector0, aVector1 );
    // }

    // /** @brief Euclidean length of input vector. */
    // inline float det( const vec2 &aVector0, const vec2 &aVector1 )
    // {
    //     return ( aVector0.x * aVector1.y ) - ( aVector0.y * aVector1.x );
    // }

    // /** @brief Unit vector in the direction of V. */
    // template <typename T>
    // T normalize( const T &aVector )
    // {
    //     return glm::normalize( aVector );
    // }

    // /** @brief Standard dot-product of v1 and v2. */
    // template <typename T>
    // inline float dot( const T &aVector1, const T &aVector2 )
    // {
    //     return glm::dot( aVector1, aVector2 );
    // }

    // /** @brief Standard cross-product of v1 and v2. */
    // inline vec3 cross( const vec3 &aVector1, const vec3 &aVector2 ) { return glm::cross( aVector1, aVector2 ); }

    // /** @brief Returns a vector normal to v1. */
    // inline vec2 perpendicular( const vec2 &aVector1 ) { return { -aVector1.y, aVector1.x }; }

    public class Functions
    {
        static float sin(float x) { return (float) System.Math.Sin((double)x ); }

        static float asin(float x) { return (float) System.Math.Asin((double)x ); }

        static float cos(float x) { return (float) System.Math.Cos((double)x ); }

        static float acos(float x) { return (float) System.Math.Acos((double)x ); }
    }

}