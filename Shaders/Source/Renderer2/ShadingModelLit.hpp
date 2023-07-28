#ifndef _LIGHT_CALCULATION_H_
#define _LIGHT_CALCULATION_H_

#if defined( __cplusplus )
#    include "Common/Definitions.h"
#    include "FragmentShaderUniformInputs.hpp"
#    include "LightData.hpp"
#    include "Material.hpp"
#    include "ShadingData.hpp"
#    include "ShadingModelLit.hpp"
#    include "SurfaceShadingStandard.hpp"
#    include "Varying.hpp"
#endif

void AddEmissive( MaterialInputs aMaterial, inout float4 aColor )
{
#if defined( MATERIAL_HAS_EMISSIVE )
    float2 lEmissive    = material.mEmissive;
    float  lAttenuation = mix( 1.0, getExposure(), lEmissive.w );
    aColor.rgb += lEmissive.rgb * ( lAttenuation * aColor.a );
#endif
}

void AddAmbient( MaterialInputs aMaterial, inout float4 aColor )
{
    aColor.rgb += 0.03 * aMaterial.mBaseColor.xyz;
}

float ComputeDiffuseAlpha( float a )
{
#if defined( BLEND_MODE_TRANSPARENT ) || defined( BLEND_MODE_FADE )
    return a;
#else
    return 1.0;
#endif
}

// For future implementation
void EvaluateIBL( MaterialInputs aMaterial, ShadingData aShadingData, float3 inWorldPos, inout float3 aColor )
{
}

void EvaluateDirectionalLight( MaterialInputs aMaterial, ShadingData aShadingData, float3 inWorldPos, inout float3 aColor )
{
}

void ComputePointLightData( float3 inWorldPos, float3 aSurfaceNormal, float3 aEyeDirection, // samplerCube aShadowMap,
                            sPunctualLight aInData, out LightData aLightData )
{
    aLightData.mColorIntensity = aInData.mColorIntensity; //( aInData.Color, aInData.Intensity );
    aLightData.mL              = normalize( aInData.mPosition - inWorldPos );
    aLightData.mH              = normalize( aEyeDirection + aLightData.mL );
    aLightData.mNdotL          = clamp( dot( aSurfaceNormal, aLightData.mL ), 0.0, 1.0 );
    aLightData.mWorldPosition  = aInData.mPosition;
    aLightData.mVisibility     = 1.0;

    float3 v                = aLightData.mWorldPosition - inWorldPos;
    float  lDistanceSqared  = dot( v, v );
    aLightData.mAttenuation = 1.0 / max( lDistanceSqared, 1e-4 );

    // float3 coords = -aLightData.mL;
    // float  lShadowDistanceSquared = texture( aShadowMap, coords ).r;
    // aLightData.mVisibility = ( lDistanceSqared <= lShadowDistanceSquared + EPSILON ) ? 1.0 : 0.0;
}

void EvaluatePunctualLights( MaterialInputs aMaterial, ShadingData aShadingData, float3 inWorldPos, inout float3 aColor )
{
    float3 V = normalize( gCamera.mPosition - inWorldPos );

    for( int i = 0; i < gPunctualLights.mArray.length(); i++ )
    {
        LightData lLightData;

        ComputePointLightData( inWorldPos, aMaterial.mNormal, V, gPunctualLights.mArray[i], lLightData );

#if defined( MATERIAL_HAS_CURTOM_SURFACE_SHADING )
        aColor.rgb += CurtomSurfaceShading( V, aMaterial.mNormal, aShadingData, lLightData )
#else
        aColor.rgb += SurfaceShading( V, aMaterial.mNormal, aShadingData, lLightData );
#endif
    }
}

#define MIN_PERCEPTUAL_ROUGHNESS 0.045
#define MIN_ROUGHNESS            0.002025
void ComputeShadingData( float3 aBaseColor, float3 aReflectance, float aMetal, float aRough, out ShadingData aShadingData )
{
    aShadingData.mF0                  = aBaseColor * aMetal + ( aReflectance * ( 1.0 - aMetal ) );
    aShadingData.mPerceptualRoughness = clamp( aRough, MIN_PERCEPTUAL_ROUGHNESS, 1.0 );
    aShadingData.mRoughness           = aShadingData.mPerceptualRoughness * aShadingData.mPerceptualRoughness;
    aShadingData.mDiffuseColor        = ( 1.0 - aMetal ) * aBaseColor;
}

float4 EvaluateLights( MaterialInputs aMaterial )
{
    float3 lColor = float3( 0.0 ); // aMaterial.mBaseColor.xyz;

    ShadingData lShadingData;
    ComputeShadingData( aMaterial.mBaseColor.rgb * aMaterial.mBaseColor.a, float3( 0.04 ), aMaterial.mMetallic, aMaterial.mRoughness,
                        lShadingData );

    EvaluateIBL( aMaterial, lShadingData, inPos, lColor );
    EvaluateDirectionalLight( aMaterial, lShadingData, inPos, lColor );
    EvaluatePunctualLights( aMaterial, lShadingData, inPos, lColor );

    return float4( lColor, ComputeDiffuseAlpha( aMaterial.mBaseColor.a ) );
}

float4 EvaluateMaterial( MaterialInputs aMaterial )
{
    float4 lColor = EvaluateLights( aMaterial );
    AddAmbient( aMaterial, lColor );
    AddEmissive( aMaterial, lColor );

    return lColor;
}

#endif
