#ifndef _DIRECTIONAL_LIGHTING_H_
#define _DIRECTIONAL_LIGHTING_H_

#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "LightData.hpp"
#    include "LightingCommon.hpp"
#    include "Varying.hpp"
#endif

float3 GetWorldPosition()
{
    return inWorldPos;
}

float3 GetCameraPosition()
{
    return gCamera.mPosition;
}

sDirectionalLight GetDirectionalLightData()
{
    return gDirectionalLight.mData;
}

LightData GetDirectionalLight()
{
    sDirectionalLight lDirectionalLightData = GetDirectionalLightData();

    LightData lLightData;

    lLightData.mColorIntensity = lDirectionalLightData.mColorIntensity;
    lLightData.mL              = normalize( lDirectionalLightData.mDirection );
    lLightData.mH              = normalize( GetEyeDirection() + lLightData.mL );
    lLightData.mNdotL          = clamp( dot( GetSurfaceNormal(), lLightData.mL ), 0.0, 1.0 );
    lLightData.mWorldPosition  = float3( 0.0 );
    lLightData.mAttenuation    = 1.0;

    return lLightData;
}


void EvaluateDirectionalLight( MaterialInputs aMaterial, ShadingData aShadingData, float3 inWorldPos, inout float3 aColor )
{
    LightData lLightData = GetDirectionalLight();

    float lVisibility = Shadow( true, gDirectionalLightShadowMap );

#if defined( MATERIAL_HAS_AMBIENT_OCCLUSION )
    lVisibility *= ComputeMicroShadowing( lLightData.mNdotL, aMaterial.mAmbientOcclusion );
#endif

    if( lVisibility <= 0.0 )
    {
        return;
    }

#if defined( MATERIAL_HAS_CURTOM_SURFACE_SHADING )
    aColor.rgb += CurtomSurfaceShading( GetEyeDirection(), aMaterial.mNormal, aShadingData, lLightData )
#else
    aColor.rgb += SurfaceShading( GetEyeDirection(), aMaterial.mNormal, aShadingData, lLightData );
#endif
}

#endif
