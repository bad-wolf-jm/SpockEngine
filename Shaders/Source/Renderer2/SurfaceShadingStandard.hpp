#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

float3 SurfaceShading( float3 aSurfaceNormal, float3 aEyeDirection, ShadingData aShadingData, LightData aLightData )
{
    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the base color as F0 (metallic workflow)
    // float3 lF0 = mix( float3( 0.04 ), aBaseColor, aMetal );
    float3 lRadiance = aLightData.mColorIntensity.xyz * aLightData.mColorIntensity.w * aLightData.mAttenuation;

    float3 H = normalize( aEyeDirection + aLightData.mL );

    // Cook-Torrance BRDF
    float3 lSpecular = CookTorrance( aShadingData.mF0, aSurfaceNormal, aLightData.mL, aEyeDirection, H, aShadingData.mRoughness );

    // kS is equal to Fresnel
    float3 kS = FresnelSchlick( max( dot( H, aEyeDirection ), 0.0 ), aShadingData.mF0 );

    // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
    // this relationship the diffuse component (kD) should equal 1.0 - kS.
    float3 kD = float3( 1.0 ) - kS;

    // add to outgoing radiance Lo
    return ( kD * aShadingData.mDiffuseColor / PI + lSpecular ) * lRadiance * aLightData.mNdotL * aLightData.mVisibility;
}