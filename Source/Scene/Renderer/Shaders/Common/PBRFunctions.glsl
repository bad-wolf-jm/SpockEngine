#define MIN_PERCEPTUAL_ROUGHNESS 0.045
#define MIN_ROUGHNESS            0.002025

struct ShadingData
{
    // The material's diffuse color, as derived from baseColor and metallic.
    // This color is pre-multiplied by alpha and in the linear sRGB color space.
    vec3  mDiffuseColor;

    // The material's specular color, as derived from baseColor and metallic.
    // This color is pre-multiplied by alpha and in the linear sRGB color space.
    vec3  mF0;

    // The perceptual roughness is the roughness value set in MaterialInputs,
    // with extra processing:
    // - Clamped to safe values
    // - Filtered if specularAntiAliasing is enabled
    // This value is between 0.0 and 1.0.
    float mPerceptualRoughness;

    // The roughness value expected by BRDFs. This value is the square of
    // perceptualRoughness. This value is between 0.0 and 1.0.
    float mRoughness;    
};

void ComputeShadingData(vec3 aBaseColor, vec3 aReflectance, float aMetal, float aRough, inout ShadingData aShadingData)
{
    aShadingData.mF0 = aBaseColor * aMetal + (aReflectance * (1.0 - aMetal));
    aShadingData.mPerceptualRoughness = clamp(aRough, 0.0, 1.0);
    aShadingData.mRoughness = aShadingData.mPerceptualRoughness * aShadingData.mPerceptualRoughness;
    aShadingData.mDiffuseColor =  ( 1.0 - aMetal ) * aBaseColor; 
}

// ----------------------------------------------------------------------------
float DistributionGGX( vec3 N, vec3 H, float roughness )
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
float GeometrySmith( vec3 N, vec3 V, vec3 L, float roughness )
{
    float NdotV = max( dot( N, V ), 0.0 );
    float NdotL = max( dot( N, L ), 0.0 );
    float ggx2  = GeometrySchlickGGX( NdotV, roughness );
    float ggx1  = GeometrySchlickGGX( NdotL, roughness );

    return ggx1 * ggx2;
}

// ----------------------------------------------------------------------------
vec3 FresnelSchlick( float cosTheta, vec3 F0 ) 
{ 
    return F0 + ( vec3( 1.0 ) - F0 ) * pow( clamp( 1.0 - cosTheta, 0.0, 1.0 ), 5.0 ); 
}

vec3 CookTorrance( vec3 F0, vec3 N, vec3 L, vec3 V, vec3 H, float roughness )
{
    // Cook-Torrance BRDF
    float NDF = DistributionGGX( N, H, roughness );
    float G   = GeometrySmith( N, V, L, roughness );
    vec3  F   = FresnelSchlick( max( dot( H, V ), 0.0 ), F0 );
    return ( NDF * G * F ) / ( 4 * max( dot( N, V ), 0.0 ) * max( dot( N, L ), 0.0 ) + 0.0001 );
}

vec3 ComputeLightContribution( vec3 aSurfaceNormal, vec3 aEyeDirection, ShadingData aShadingData, LightData aLightData )
{
    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the base color as F0 (metallic workflow)
    // vec3 lF0 = mix( vec3( 0.04 ), aBaseColor, aMetal );
    vec3 lRadiance = aLightData.mColorIntensity.xyz * aLightData.mColorIntensity.w * aLightData.mAttenuation;

    vec3 H = normalize( aEyeDirection + aLightData.mL );

    // Cook-Torrance BRDF
    vec3 lSpecular = CookTorrance( aShadingData.mF0, aSurfaceNormal, aLightData.mL, aEyeDirection, H, aShadingData.mRoughness );

    // kS is equal to Fresnel
    vec3 kS = FresnelSchlick( max( dot( H, aEyeDirection ), 0.0 ), aShadingData.mF0 );

    // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
    // this relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3( 1.0 ) - kS;

    // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal
    // (pure metals have no diffuse light).
    // kD *= ( 1.0 - aMetal );

    // scale light by NdotL
    //float NdotL = max( dot( aSurfaceNormal, aLightDirection ), 0.0 );

    // add to outgoing radiance Lo
    return ( kD * aShadingData.mDiffuseColor / PI + lSpecular ) * lRadiance * aLightData.mNdotL * aLightData.mVisibility;
}

