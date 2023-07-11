#version 460
#extension GL_EXT_nonuniform_qualifier : enable

layout( location = 0 ) in vec3 inWorldPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;

#define MAX_NUM_LIGHTS 64

struct DirectionalLightData
{
    vec3  Direction;
    vec3  Color;
    float Intensity;
};

struct PointLightData
{
    vec3  WorldPosition;
    vec3  Color;
    float Intensity;
};

struct SpotlightData
{
    vec3  WorldPosition;
    vec3  LookAtDirection;
    vec3  Color;
    float Intensity;
    float Cone;
};

struct sShaderMaterial
{
    vec4 mBaseColorFactor;
    int  mBaseColorTextureID;
    int  mBaseColorUVChannel;

    float mMetallicFactor;
    float mRoughnessFactor;
    int   mMetalnessUVChannel;
    int   mMetalnessTextureID;

    float mOcclusionStrength;
    int   mOcclusionUVChannel;
    int   mOcclusionTextureID;

    vec4 mEmissiveFactor;
    int  mEmissiveTextureID;
    int  mEmissiveUVChannel;

    int mNormalTextureID;
    int mNormalUVChannel;

    float mAlphaThreshold;
};

// Scene bindings
layout( set = 0, binding = 0 ) uniform UBO
{
    mat4 projection;
    mat4 model;
    mat4 view;
    vec3 camPos;

    int                  DirectionalLightCount;
    DirectionalLightData DirectionalLights[MAX_NUM_LIGHTS];

    int           SpotlightCount;
    SpotlightData Spotlights[MAX_NUM_LIGHTS];

    int            PointLightCount;
    PointLightData PointLights[MAX_NUM_LIGHTS];
}
ubo;

layout( set = 0, binding = 1 ) uniform UBOParams
{
    float exposure;
    float gamma;
    float AmbientLightIntensity;
    vec4  AmbientLightColor;
    float debugViewInputs;
    float debugViewEquation;
    float grayscaleRendering;
}
uboParams;

layout( std140, set = 1, binding = 0 ) readonly buffer ShaderMaterial { sShaderMaterial mArray[]; }
gMaterials;

layout( set = 1, binding = 1 ) uniform sampler2D gTextures[];

layout( push_constant ) uniform Material { uint mMaterialID; }
material;

layout( location = 0 ) out vec4 outColor;

const float PI             = 3.14159265359;
const float M_PI           = 3.141592653589793;
const float c_MinRoughness = 0.04;

vec4 SRGBtoLINEAR( vec4 srgbIn )
{
    vec3 bLess  = step( vec3( 0.04045 ), srgbIn.xyz );
    vec3 linOut = mix( srgbIn.xyz / vec3( 12.92 ), pow( ( srgbIn.xyz + vec3( 0.055 ) ) / vec3( 1.055 ), vec3( 2.4 ) ), bLess );
    return vec4( linOut, srgbIn.w );
}

vec3 getNormalFromMap( sampler2D aNormalSampler, vec2 aCoords )
{
    // // Perturb normal, see http://www.thetenthplanet.de/archives/1180
    // if( material.normalTextureSet == -1.0 )
    //     return normalize( inNormal );

    vec3 tangentNormal = texture( aNormalSampler, aCoords ).xyz * 2.0 - vec3( 1.0 );

    vec3 q1  = dFdx( inWorldPos );
    vec3 q2  = dFdy( inWorldPos );
    vec2 st1 = dFdx( inUV0 );
    vec2 st2 = dFdy( inUV0 );

    vec3 N   = normalize( inNormal );
    vec3 T   = normalize( q1 * st2.t - q2 * st1.t );
    vec3 B   = -normalize( cross( N, T ) );
    mat3 TBN = mat3( T, B, N );

    return normalize( TBN * tangentNormal );
}

#include "Common/ToneMap.glsli"
#include "Common/PBRFunctions.glsli"

void main()
{
    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    vec4 lBaseColor = lMaterial.mBaseColorFactor;
    lBaseColor =
        SRGBtoLINEAR( texture( gTextures[lMaterial.mBaseColorTextureID], inUV0 ) ) *
        lMaterial.mBaseColorFactor;

    float metallic      = clamp( lMaterial.mMetallicFactor, 0.0, 1.0 );
    float roughness     = clamp( lMaterial.mRoughnessFactor, c_MinRoughness, 1.0 );
    vec4  lSampledValue = texture( gTextures[lMaterial.mMetalnessTextureID], inUV0 );
    metallic            = lSampledValue.r * lMaterial.mMetallicFactor;
    roughness           = lSampledValue.g * lMaterial.mRoughnessFactor;

    vec3 emissive = vec3( lMaterial.mEmissiveFactor );
    emissive =
        SRGBtoLINEAR( texture( gTextures[lMaterial.mEmissiveTextureID], inUV0 ) ).rgb *
        vec3( lMaterial.mEmissiveFactor );

    vec3 N;
    if( lMaterial.mNormalTextureID == 0 )
        N = normalize( inNormal );
    else
        N = getNormalFromMap( gTextures[lMaterial.mNormalTextureID], inUV0 );

    vec3 V = normalize( ubo.camPos - inWorldPos );

    // reflectance equation
    vec3 Lo = vec3( 0.0f );

    for( int lDirectionalLightIndex = 0; lDirectionalLightIndex < ubo.DirectionalLightCount; lDirectionalLightIndex++ )
    {
        vec3 radiance = ubo.DirectionalLights[lDirectionalLightIndex].Color * ubo.DirectionalLights[lDirectionalLightIndex].Intensity;
        vec3 lLightDirection = normalize( ubo.DirectionalLights[lDirectionalLightIndex].Direction );
        Lo += ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, radiance, metallic, roughness );
    }

    for( int lPointLightIndex = 0; lPointLightIndex < ubo.PointLightCount; lPointLightIndex++ )
    {
        vec3 lLightPosition  = ubo.PointLights[lPointLightIndex].WorldPosition;
        vec3 lLightDirection = normalize( lLightPosition - inWorldPos );

        vec3 lRadiance = ComputeRadiance( ubo.PointLights[lPointLightIndex].WorldPosition, inWorldPos,
                                          ubo.PointLights[lPointLightIndex].Color, ubo.PointLights[lPointLightIndex].Intensity );
        Lo += ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, lRadiance, metallic, roughness );
    }

    for( int lSpotlightIndex = 0; lSpotlightIndex < ubo.SpotlightCount; lSpotlightIndex++ )
    {
        vec3  L                   = normalize( ubo.Spotlights[lSpotlightIndex].WorldPosition - inWorldPos );
        vec3  lLightDirection     = normalize( ubo.Spotlights[lSpotlightIndex].LookAtDirection );
        float lAngleToLightOrigin = dot( L, normalize( -lLightDirection ) );

        if( lAngleToLightOrigin < ubo.Spotlights[lSpotlightIndex].Cone ) continue;

        vec3 lRadiance = ComputeRadiance( ubo.Spotlights[lSpotlightIndex].WorldPosition, inWorldPos,
                                          ubo.Spotlights[lSpotlightIndex].Color, ubo.Spotlights[lSpotlightIndex].Intensity );

        Lo += ComputeLightContribution( lBaseColor.xyz, N, V, L, lRadiance, metallic, roughness );
    }

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient   = uboParams.AmbientLightIntensity * uboParams.AmbientLightColor.rgb * lBaseColor.xyz;
    vec3 hdr_color = ambient + Lo;

    float ao  = texture( gTextures[lMaterial.mOcclusionTextureID], inUV0 ).r;
    hdr_color = mix( hdr_color, hdr_color * ao, lMaterial.mOcclusionStrength );

    hdr_color = hdr_color + emissive;
    // vec3 hdr_color = Lo;
    vec4 fullcolor = vec4( hdr_color, lBaseColor.a );

    outColor = tonemap( fullcolor );
    if( uboParams.grayscaleRendering == 1.0f ) outColor = vec4( vec3( dot( outColor.xyz, vec3( 0.2126, 0.7152, 0.0722 ) ) ), lBaseColor.a );
    // outColor = vec4(vec3((length(inWorldPos) - 0.5)), 1.0);
}