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

#if defined( MATERIAL_HAS_SHEEN_COLOR )
float3 SheenLobe( const ShadingData pixel, float NoV, float NoL, float NoH )
{
    float D = D_Charlie( pixel.mSheenRoughness, NoH );
    float V = V_Neubelt( NoV, NoL );

    return ( D * V ) * pixel.sheenColor;
}
#endif

#if defined( MATERIAL_HAS_CLEAR_COAT )
float ClearCoatLobe( const ShadingData pixel, const float3 h, float NoH, float LoH, out float Fcc )
{
#    if defined( MATERIAL_HAS_NORMAL ) || defined( MATERIAL_HAS_CLEAR_COAT_NORMAL )
    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    float clearCoatNoH = saturate( dot( shading_clearCoatNormal, h ) );
#    else
    float clearCoatNoH = NoH;
#    endif

    // clear coat specular lobe
    float D = Distribution( pixel.mClearCoatRoughness, clearCoatNoH, h );
    float V = V_Kelemen( LoH );
    float F = F_Schlick( 0.04, 1.0, LoH ) * pixel.mClearCoat; // fix IOR to 1.5

    Fcc = F;
    return D * V * F;
}
#endif

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

#if defined( MATERIAL_HAS_SHEEN_COLOR )
    color *= aShadingData.mSheenScaling;
    color += SheenLobe( aShadingData, NoV, NoL, NoH );
#endif

#if defined( MATERIAL_HAS_CLEAR_COAT )
    float Fcc;
    float clearCoat   = ClearCoatLobe( aShadingData, h, NoH, LoH, Fcc );
    float attenuation = 1.0 - Fcc;
#    if defined( MATERIAL_HAS_NORMAL ) || defined( MATERIAL_HAS_CLEAR_COAT_NORMAL )
    color *= attenuation * NoL;

    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    float clearCoatNoL = saturate( dot( shading_clearCoatNormal, light.l ) );
    color += clearCoat * clearCoatNoL;

    // Early exit to avoid the extra multiplication by NoL
    return ( color * light.colorIntensity.rgb ) * ( light.colorIntensity.w * light.attenuation * occlusion );
#    else
    color *= attenuation;
    color += clearCoat;
#    endif
#endif

    return ( lColor * aLightData.mColorIntensity.rgb ) *
           ( aLightData.mColorIntensity.w * aLightData.mAttenuation * NdotL * aLightData.mVisibility );
}
