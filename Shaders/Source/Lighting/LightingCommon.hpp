#ifndef _LIGHTING_COMMON_H
#define _LIGHTING_COMMON_H


#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "LightData.hpp"
#    include "Varying.hpp"
#endif

float3 GetEyeDirection()
{
    return normalize( gCamera.mPosition - GetWorldPosition() );
}

float ComputeMicroShadowing( float aNdotL, float aAmbientOcclusion )
{
    // Chan 2018, "Material Advances in Call of Duty: WWII"
    float lAperture    = inversesqrt( 1.0 - min( aAmbientOcclusion, 0.9999 ) );
    float lMicroShadow = clamp( aNdotL * lAperture, 0.0, 1.0 );

    return lMicroShadow * lMicroShadow;
}

#endif