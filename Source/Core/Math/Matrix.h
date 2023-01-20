/// @file   Matrix.h
///
/// @brief  Abstraction for glm's matrix types
///
/// @author Jean-Martin Albert
///
/// @copyright (c) 2021 LeddarTech Inc. All rights reserved.

#pragma once

#include "Vector.h"
#include "glm.h"

namespace math
{
    // Matrix types
    using mat3 = glm::mat3;
    using mat4 = glm::mat4;

    template <typename _Ty> inline _Ty TwoPi() { return glm::two_pi<_Ty>(); }

    inline mat4 MakeMat4( const float *aEntries ) { return glm::make_mat4( aEntries ); }

    inline mat3 MakeMat3( const float *aEntries ) { return glm::make_mat3( aEntries ); }

    inline mat4 LookAt( vec3 const &aEye, vec3 const &aCenter, vec3 const &aUp ) { return glm::lookAt( aEye, aCenter, aUp ); }

    inline mat4 Orthogonal( vec2 const& aXSpan, vec2 const& aYSpan, vec2 const& aDepth ) { return glm::ortho( aXSpan.x, aXSpan.y, aYSpan.x, aYSpan.y, aDepth.x, aDepth.y ); }

    inline mat4 Orthogonal( float aWidth, float aHeight ) { return glm::ortho( 0.f, aWidth, 0.f, aHeight, -1.f, 1.f ); }

    inline mat4 Orthogonal( vec2 const &aSize ) { return glm::ortho( 0.f, aSize.x, 0.f, aSize.y, -1.f, 1.f ); }

    inline mat4 PerspectiveLH( float aFov, float aAspect, float aNear, float aFar ) { return glm::perspectiveLH( aFov / aAspect, aAspect, aNear, aFar ); }

    inline mat4 PerspectiveRH( float aFov, float aAspect, float aNear, float aFar ) { return glm::perspectiveRH( aFov / aAspect, aAspect, aNear, aFar ); }

    inline mat4 Perspective( float aFov, float aAspect, float aNear, float aFar ) { return glm::perspective( aFov / aAspect, aAspect, aNear, aFar ); }

    inline mat4 Rotation( float aAngle, vec3 const &aAxis ) { return glm::rotate( mat4( 1.0 ), aAngle, aAxis ); }

    inline mat3 Comatrix( mat3 const &aMatrix )
    {
        mat3 lOut;

        lOut[0][0] = ( aMatrix[1][1] * aMatrix[2][2] - aMatrix[2][1] * aMatrix[1][2] );
        lOut[1][0] = -( aMatrix[0][1] * aMatrix[2][2] - aMatrix[2][1] * aMatrix[0][2] );
        lOut[2][0] = ( aMatrix[0][1] * aMatrix[1][2] - aMatrix[1][1] * aMatrix[0][2] );

        lOut[0][1] = -( aMatrix[1][0] * aMatrix[2][2] - aMatrix[2][0] * aMatrix[1][2] );
        lOut[1][1] = ( aMatrix[0][0] * aMatrix[2][2] - aMatrix[2][0] * aMatrix[0][2] );
        lOut[2][1] = -( aMatrix[0][0] * aMatrix[1][2] - aMatrix[1][0] * aMatrix[0][2] );

        lOut[0][2] = ( aMatrix[1][0] * aMatrix[2][1] - aMatrix[2][0] * aMatrix[1][1] );
        lOut[1][2] = -( aMatrix[0][0] * aMatrix[2][1] - aMatrix[2][0] * aMatrix[0][1] );
        lOut[2][2] = ( aMatrix[0][0] * aMatrix[1][1] - aMatrix[1][0] * aMatrix[0][1] );

        return lOut;
    }

    inline vec3 UpDirection( mat4 const &aMatrix ) { return vec3( aMatrix[1] ); }

    inline vec3 RightDirection( mat4 const &aMatrix ) { return vec3( aMatrix[0] ); }

    inline vec3 BackwardDirection( mat4 const &aMatrix ) { return vec3( aMatrix[2] ); }

    inline mat3 NormalMatrix( mat4 const &aMatrix ) { return Comatrix( mat3( aMatrix ) ); }

    inline mat4 FromComponents( mat3 const &aRotation, vec3 const &aTranslation )
    {
        auto lOut  = glm::mat4( aRotation );
        lOut[3][0] = aTranslation[0];
        lOut[3][1] = aTranslation[1];
        lOut[3][2] = aTranslation[2];
        return lOut;
    }

    inline mat4 FromDiagonal( vec4 const &aDiagonal )
    {
        glm::mat4 lOut( 0.0f );
        lOut[0][0] = aDiagonal[0];
        lOut[1][1] = aDiagonal[1];
        lOut[2][2] = aDiagonal[2];
        lOut[3][3] = aDiagonal[3];
        return lOut;
    }

    inline mat3 FromDiagonal( vec3 const &aDiagonal )
    {
        glm::mat3 lOut{ 0.0f };
        lOut[0][0] = aDiagonal[0];
        lOut[1][1] = aDiagonal[1];
        lOut[2][2] = aDiagonal[2];
        return lOut;
    }

    template <typename T> T Inverse( T const &a_Matrix ) { return glm::inverse( a_Matrix ); }
    template <typename T> float Determinant( T const &a_Matrix ) { return glm::determinant( a_Matrix ); }

    template <typename T> T Transpose( T const &a_Matrix ) { return glm::transpose( a_Matrix ); }

    inline mat3 Rotation( mat4 const &aMatrix )
    {
        mat3 lOut = glm::mat3( aMatrix );
        lOut[0]   = lOut[0] / length( lOut[0] );
        lOut[1]   = lOut[1] / length( lOut[1] );
        lOut[2]   = lOut[2] / length( lOut[2] );
        return lOut;
    }

    inline quat Quaternion( mat4 const &aMatrix ) { return glm::quat_cast( Rotation( aMatrix ) ); }

    inline mat4 Translate( mat4 const &aMatrix, vec3 const &aAxis ) { return glm::translate( aMatrix, aAxis ); }

    inline mat4 Scale( mat4 const &aMatrix, vec3 const &aAxis ) { return glm::scale( aMatrix, aAxis ); }

    inline mat4 Translation( vec3 const &aAxis ) { return glm::translate( mat4( 1.0 ), aAxis ); }

    inline vec3 Translation( mat4 const &aMatrix ) { return glm::vec3( aMatrix[3] ); }

    inline vec3 Scaling( mat4 const &aMatrix ) { return glm::vec3( glm::length( aMatrix[0] ), glm::length( aMatrix[1] ), glm::length( aMatrix[2] ) ); }

    inline mat4 Scaling( vec3 const &aVector ) { return Scale( mat4( 1.0 ), aVector ); }

} // namespace math
