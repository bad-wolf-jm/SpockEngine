#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif


const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX( float3 N, float3 H, float roughness )
{
    float a2     = roughness * roughness;
    float NdotH  = max( dot( N, H ), 0.0 );
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = ( NdotH2 * ( a2 - 1.0 ) + 1.0 );
    denom       = PI * denom; // * denom;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySchlickGGX( float NdotV, float roughness )
{
    float r     = ( roughness + 1.0 );
    float k     = ( r * r ) / 8.0;
    float nom   = NdotV;
    float denom = NdotV * ( 1.0 - k ) + k;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySmith( float3 N, float3 V, float3 L, float roughness )
{
    float NdotV = max( dot( N, V ), 0.0 );
    float NdotL = max( dot( N, L ), 0.0 );
    float ggx2  = GeometrySchlickGGX( NdotV, roughness );
    float ggx1  = GeometrySchlickGGX( NdotL, roughness );

    return ggx1 * ggx2;
}

// ----------------------------------------------------------------------------
float3 FresnelSchlick( float cosTheta, float3 F0 ) 
{ 
    return F0 + ( float3( 1.0 ) - F0 ) * pow( clamp( 1.0 - cosTheta, 0.0, 1.0 ), 5.0 ); 
}

float3 CookTorrance( float3 F0, float3 N, float3 L, float3 V, float3 H, float roughness )
{
    // Cook-Torrance BRDF
    float NDF = DistributionGGX( N, H, roughness );
    float G   = GeometrySmith( N, V, L, roughness );
    float3  F   = FresnelSchlick( max( dot( H, V ), 0.0 ), F0 );
    return ( NDF * G * F ) / ( 4 * max( dot( N, V ), 0.0 ) * max( dot( N, L ), 0.0 ) + 0.0001 );
}

float3 SurfaceShading( float3 V, float3 N, ShadingData aShadingData, LightData aLightData )
{
    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the base color as F0 (metallic workflow)
    // float3 lF0 = mix( float3( 0.04 ), aBaseColor, aMetal );
    float3 lRadiance = aLightData.mColorIntensity.xyz * aLightData.mColorIntensity.w * aLightData.mAttenuation;

    float3 H = normalize( V + aLightData.mL );

    // Cook-Torrance BRDF
    float3 lSpecular = CookTorrance( aShadingData.mF0, N, aLightData.mL, V, H, aShadingData.mRoughness );

    // kS is equal to Fresnel
    float3 kS = FresnelSchlick( max( dot( H, V ), 0.0 ), aShadingData.mF0 );

    // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
    // this relationship the diffuse component (kD) should equal 1.0 - kS.
    float3 kD = float3( 1.0 ) - kS;

    // add to outgoing radiance Lo
    return ( kD * aShadingData.mDiffuseColor / PI + lSpecular ) * lRadiance * aLightData.mNdotL * aLightData.mVisibility;
}