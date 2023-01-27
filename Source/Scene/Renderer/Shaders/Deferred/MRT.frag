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
}
uboParams;

layout( std140, set = 1, binding = 0 ) readonly buffer ShaderMaterial { sShaderMaterial mArray[]; }
gMaterials;

layout( set = 1, binding = 1 ) uniform sampler2D gTextures[];

layout( push_constant ) uniform Material { uint mMaterialID; }
material;

layout( location = 0 ) out vec4 outPosition;
layout( location = 1 ) out vec4 outNormal;
layout( location = 2 ) out vec4 outAlbedo;
layout( location = 3 ) out vec4 outOcclusionMetalRough;

const float c_MinRoughness = 0.04;

#include "../Common/GetNormalFromMap.glsli"

void main()
{
    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    outPosition = vec4( inWorldPos, 1.0 );

    vec3 tnorm;
    if( lMaterial.mNormalTextureID == 0 )
        tnorm = normalize( inNormal );
    else
        tnorm = getNormalFromMap( gTextures[lMaterial.mNormalTextureID], inUV0 );

    vec4  lSampledValue = texture( gTextures[lMaterial.mMetalnessTextureID], inUV0 );
    float metallic      = lSampledValue.r * clamp( lMaterial.mMetallicFactor, 0.0, 1.0 );
    float roughness     = lSampledValue.g * clamp( lMaterial.mRoughnessFactor, c_MinRoughness, 1.0 );
    float ao = texture( gTextures[lMaterial.mOcclusionTextureID], inUV0 ).r *
               lMaterial.mOcclusionStrength;
    metallic  = lSampledValue.r * lMaterial.mMetallicFactor;
    roughness = lSampledValue.g * lMaterial.mRoughnessFactor;

    outNormal = vec4( tnorm, 1.0 );
    outAlbedo = texture( gTextures[lMaterial.mBaseColorTextureID], inUV0) *
                lMaterial.mBaseColorFactor;
    outOcclusionMetalRough = vec4( ao, metallic, roughness, lMaterial.mOcclusionStrength );
}