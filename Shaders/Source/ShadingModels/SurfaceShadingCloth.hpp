#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

float Distribution( float roughness, float NoH )
{
    return D_Charlie( roughness, NoH );
}

float Visibility( float NoV, float NoL )
{
    return V_Neubelt( NoV, NoL );
}


float3 Diffuse( const ShadingData pixel, float aNoV, float aNoL, float aLoH )
{
    // Burley 2012, "Physically-Based Shading at Disney"
    float f90          = 0.5 + 2.0 * pixel.mRoughness * aLoH * aLoH;
    float lightScatter = F_Schlick( 1.0, f90, aNoL );
    float viewScatter  = F_Schlick( 1.0, f90, aNoV );

    return lightScatter * viewScatter * ( 1.0 / PI );
}

/**
 * Evaluates lit materials with the cloth shading model. Similar to the standard
 * model, the cloth shading model is based on a Cook-Torrance microfacet model.
 * Its distribution and visibility terms are however very different to take into
 * account the softer apperance of many types of cloth. Some highly reflecting
 * fabrics like satin or leather should use the standard model instead.
 *
 * This shading model optionally models subsurface scattering events. The
 * computation of these events is not physically based but can add necessary
 * details to a material.
 */
float3 SurfaceShading( float3 V, float3 N, ShadingData pixel, LightData light )
{
    float3 h     = aLightData.mH;
    float  NoL   = light.mNdotL;
    float  NoH   = saturate( dot( N, h ) );
    float  LoH   = saturate( dot( light.mL, h ) );
    float  NdotV = saturate( dot( N, V ) );

    // specular BRDF
    float  D = Distribution( pixel.mRoughness, NoH );
    float  V = Visibility( NdotV, NoL );
    float3 F = pixel.mF0;

    // Ignore pixel.energyCompensation since we use a different BRDF here
    float3 Fr = ( D * V ) * F;

    // diffuse BRDF
    float diffuse = Diffuse( pixel.mRoughness, NdotV, NoL, LoH );

#if defined( MATERIAL_HAS_SUBSURFACE_COLOR )
    // Energy conservative wrap diffuse to simulate subsurface scattering
    diffuse *= Fd_Wrap( dot( N, light.mL ), 0.5 );
#endif

    // We do not multiply the diffuse term by the Fresnel term as discussed in
    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    // The effect is fairly subtle and not deemed worth the cost for mobile
    float3 Fd = diffuse * pixel.mDiffuseColor;

#if defined( MATERIAL_HAS_SUBSURFACE_COLOR )
    // Cheap subsurface scatter
    Fd *= saturate( pixel.mSubSurfaceColor + NoL );
    // We need to apply NoL separately to the specular lobe since we already took
    // it into account in the diffuse lobe
    float3 color = Fd + Fr * NoL;
    color *= light.mColorIntensity.rgb * ( light.mColorIntensity.w * light.mAttenuation * light.mVisibility );
#else
    float3 color = Fd + Fr;
    color *= light.mColorIntensity.rgb * ( light.mColorIntensity.w * light.mAttenuation * NoL * light.mVisibility );
#endif

    return color;
}
