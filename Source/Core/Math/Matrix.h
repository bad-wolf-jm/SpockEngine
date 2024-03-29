/// @file   Matrix.h
///
/// @brief  Abstraction for glm's matrix types
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2023 Jean-Martin Albert All rights reserved.

#pragma once

#include "Vector.h"
#include "glm.h"

namespace math
{
    // Matrix types
    using mat3 = glm::mat3;
    using mat4 = glm::mat4;

    template <typename _Ty>
    inline _Ty TwoPi()
    {
        return glm::two_pi<_Ty>();
    }

    inline mat4 look_at( vec3 const &eye, vec3 const &center, vec3 const &up )
    {
        return glm::lookAt( eye, center, up );
    }

    inline mat4 orthogonal( vec2 const &xSpan, vec2 const &ySpan, vec2 const &depth )
    {
        return glm::ortho( xSpan.x, xSpan.y, ySpan.x, ySpan.y, depth.x, depth.y );
    }

    inline mat4 orthogonal( float width, float height )
    {
        return glm::ortho( 0.f, width, 0.f, height, -1.f, 1.f );
    }

    inline mat4 orthogonal( vec2 const &size )
    {
        return glm::ortho( 0.f, size.x, 0.f, size.y, -1.f, 1.f );
    }

    // inline mat4 perspective_lh( float fov, float aspect, float nearDistance, float farDistance )
    // {
    //     return glm::perspectiveLH( fov / aspect, aspect, nearDistance, farDistance );
    // }

    // inline mat4 PerspectiveRH( float fov, float aspect, float nearDistance, float farDistance )
    // {
    //     return glm::perspectiveRH( fov / aspect, aspect, nearDistance, farDistance );
    // }

    inline mat4 perspective( float fov, float aspect, float nearDistance, float farDistance )
    {
        return glm::perspective( fov / aspect, aspect, nearDistance, farDistance );
    }

    inline mat4 rotation( float angle, vec3 const &axis )
    {
        return glm::rotate( mat4( 1.0 ), angle, axis );
    }

    inline mat3 comatrix( mat3 const &matrix )
    {
        mat3 out;

        out[0][0] = ( matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2] );
        out[1][0] = -( matrix[0][1] * matrix[2][2] - matrix[2][1] * matrix[0][2] );
        out[2][0] = ( matrix[0][1] * matrix[1][2] - matrix[1][1] * matrix[0][2] );

        out[0][1] = -( matrix[1][0] * matrix[2][2] - matrix[2][0] * matrix[1][2] );
        out[1][1] = ( matrix[0][0] * matrix[2][2] - matrix[2][0] * matrix[0][2] );
        out[2][1] = -( matrix[0][0] * matrix[1][2] - matrix[1][0] * matrix[0][2] );

        out[0][2] = ( matrix[1][0] * matrix[2][1] - matrix[2][0] * matrix[1][1] );
        out[1][2] = -( matrix[0][0] * matrix[2][1] - matrix[2][0] * matrix[0][1] );
        out[2][2] = ( matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1] );

        return out;
    }

    inline vec3 up_direction( mat4 const &matrix )
    {
        return vec3( matrix[1] );
    }

    inline vec3 right_direction( mat4 const &matrix )
    {
        return vec3( matrix[0] );
    }

    inline vec3 backward_direction( mat4 const &matrix )
    {
        return vec3( matrix[2] );
    }

    inline mat3 normal_matrix( mat4 const &matrix )
    {
        return comatrix( mat3( matrix ) );
    }

    inline mat4 from_components( mat3 const &rotation, vec3 const &translation )
    {
        auto out  = glm::mat4( rotation );
        out[3][0] = translation[0];
        out[3][1] = translation[1];
        out[3][2] = translation[2];
        return out;
    }

    inline mat4 from_diagonal( vec4 const &diagonal )
    {
        glm::mat4 out( 0.0f );
        out[0][0] = diagonal[0];
        out[1][1] = diagonal[1];
        out[2][2] = diagonal[2];
        out[3][3] = diagonal[3];
        return out;
    }

    inline mat3 from_diagonal( vec3 const &diagonal )
    {
        glm::mat3 out{ 0.0f };
        out[0][0] = diagonal[0];
        out[1][1] = diagonal[1];
        out[2][2] = diagonal[2];
        return out;
    }

    template <typename T>
    T inverse( T const &matrix )
    {
        return glm::inverse( matrix );
    }

    template <typename T>
    float determinant( T const &matrix )
    {
        return glm::determinant( matrix );
    }

    template <typename T>
    T transpose( T const &matrix )
    {
        return glm::transpose( matrix );
    }

    inline mat3 rotation( mat4 const &matrix )
    {
        mat3 out = glm::mat3( matrix );
        out[0]   = out[0] / length( out[0] );
        out[1]   = out[1] / length( out[1] );
        out[2]   = out[2] / length( out[2] );
        return out;
    }

    inline quat quaternion( mat4 const &matrix )
    {
        return glm::quat_cast( rotation( matrix ) );
    }

    inline mat4 translate( mat4 const &matrix, vec3 const &axis )
    {
        return glm::translate( matrix, axis );
    }

    inline mat4 scale( mat4 const &matrix, vec3 const &axis )
    {
        return glm::scale( matrix, axis );
    }

    inline mat4 translation( vec3 const &axis )
    {
        return glm::translate( mat4( 1.0 ), axis );
    }

    inline vec3 translation( mat4 const &matrix )
    {
        return glm::vec3( matrix[3] );
    }

    inline vec3 scaling( mat4 const &matrix )
    {
        return glm::vec3( glm::length( matrix[0] ), glm::length( matrix[1] ), glm::length( matrix[2] ) );
    }

    inline mat4 scaling( vec3 const &vector )
    {
        return scale( mat4( 1.0 ), vector );
    }

} // namespace math
