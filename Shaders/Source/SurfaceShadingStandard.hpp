#if defined( __cplusplus )
#    include "Brdf.hpp"
#    include "Common/Definitions.hpp"
#    include "Common/HelperFunctions.hpp"
#endif

float Distribution( float roughness, float NoH, const float3 h )
{
    return D_GGX( roughness, NoH, h );
}

float Visibility( float roughness, float NoV, float NoL )
{
    return V_SmithGGXCorrelated( roughness, NoV, NoL );
}

float3 Fresnel( const float3 f0, float LoH )
{
    float f90 = saturate( dot( f0, float3( 50.0 * 0.33 ) ) );

    return F_Schlick( f0, f90, LoH );
}

float3 IsotropicLobe( const ShadingData pixel, const LightData light, const float3 h, float NoV, float NoL, float NoH, float LoH )
{
    float  D = Distribution( pixel.mRoughness, NoH, h );
    float  V = Visibility( pixel.mRoughness, NoV, NoL );
    float3 F = Fresnel( pixel.mF0, LoH );

    return ( D * V ) * F;
}

float3 SpecularLobe( ShadingData aShadingData, LightData aLightData, float aNdotV, float aNdotL, float aNdotH, float aLdotH )
{
#if defined( MATERIAL_HAS_ANISOTROPY )
    return AnisotropicLobe( aShadingData, aLightData, aLightData.mH, aNdotV, aNdotL, aNdotH, aLdotH );
#else
    return IsotropicLobe( aShadingData, aLightData, aLightData.mH, aNdotV, aNdotL, aNdotH, aLdotH );
#endif
}

float3 DiffuseLobe( const ShadingData pixel, float aNoV, float aNoL, float aLoH )
{
    // Burley 2012, "Physically-Based Shading at Disney"
    float f90          = 0.5 + 2.0 * pixel.mRoughness * aLoH * aLoH;
    float lightScatter = F_Schlick( 1.0, f90, aNoL );
    float viewScatter  = F_Schlick( 1.0, f90, aNoV );

    return pixel.mDiffuseColor * lightScatter * viewScatter * ( 1.0 / PI );
}

// Implement the standard shading model from Filament
float3 SurfaceShading( float3 V, float3 N, ShadingData aShadingData, LightData aLightData )
{
    float3 H = aLightData.mH;

    float NdotV = saturate( dot( N, V ) );
    float NdotL = aLightData.mNdotL;
    float NdotH = saturate( dot( N, H ) );
    float LdotH = saturate( dot( aLightData.mL, H ) );

    float3 Fr = SpecularLobe( aShadingData, aLightData, NdotV, NdotL, NdotH, LdotH );
    float3 Fd = DiffuseLobe( aShadingData, NdotV, NdotL, NdotH );

    float3 lColor = ( Fr * aShadingData.mEnergyCompensation ) + Fd;

    // TODO: Sheen

    // TODO: Clear coat

    return ( lColor * aLightData.mColorIntensity.rgb ) *
           ( aLightData.mColorIntensity.w * aLightData.mAttenuation * NdotL * aLightData.mVisibility );
}
