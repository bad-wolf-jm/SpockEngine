#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "Common/HelperFunctions.hpp"
#endif

float D_GGX( float roughness, float NoH, const float3 h )
{
    // Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"

    float oneMinusNoHSquared = 1.0 - NoH * NoH;

    float a = NoH * roughness;
    float k = roughness / ( oneMinusNoHSquared + a * a );
    float d = k * k * ( 1.0 / PI );

    return saturate( d );
}

float Distribution( float roughness, float NoH, const float3 h )
{
    return D_GGX( roughness, NoH, h );
}

float V_SmithGGXCorrelated( float roughness, float NoV, float NoL )
{
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    float a2 = roughness * roughness;

    // TODO: lambdaV can be pre-computed for all the lights, it should be moved out of this function
    float lambdaV = NoL * sqrt( ( NoV - a2 * NoV ) * NoV + a2 );
    float lambdaL = NoV * sqrt( ( NoL - a2 * NoL ) * NoL + a2 );
    float v       = 0.5 / ( lambdaV + lambdaL );

    return saturate( v );
}

float Visibility( float roughness, float NoV, float NoL )
{
    return V_SmithGGXCorrelated( roughness, NoV, NoL );
}

float3 F_Schlick( const float3 f0, float f90, float VoH )
{
    // Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
    return f0 + ( f90 - f0 ) * pow5( 1.0 - VoH );
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

float F_Schlick( float f0, float f90, float VoH )
{
    return f0 + ( f90 - f0 ) * pow5( 1.0 - VoH );
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

    float3 lColor = (Fr * aShadingData.mEnergyCompensation) + Fd;

    // TODO: Sheen

    // TODO: Clear coat

    return ( lColor * aLightData.mColorIntensity.rgb ) *
           ( aLightData.mColorIntensity.w * aLightData.mAttenuation * NdotL * aLightData.mVisibility );
}
