#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// layout( binding = 1 ) uniform sampler2D samplerColor;
// layout( binding = 2 ) uniform sampler2D samplerNormalMap;

layout( location = 0 ) in vec3 inWorldPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;
layout( location = 3 ) in vec2 inUV1;

// layout (location = 0) in vec3 inNormal;
// layout (location = 1) in vec2 inUV;
// layout (location = 2) in vec3 inColor;
// layout (location = 3) in vec3 inWorldPos;
// layout( location = 4 ) in vec3 inTangent;

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
layout( location = 3 ) out vec4 outSpecular;

void main()
{
    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    outPosition = vec4( inWorldPos, 1.0 );

    vec3 tangentNormal =
        texture( gTextures[lMaterial.mNormalTextureID], lMaterial.mNormalUVChannel == 0 ? inUV0 : inUV1 ).xyz * 2.0 - vec3( 1.0 );

    vec3 q1  = dFdx( inWorldPos );
    vec3 q2  = dFdy( inWorldPos );
    vec2 st1 = dFdx( lMaterial.mNormalUVChannel == 0 ? inUV0 : inUV1 );
    vec2 st2 = dFdy( lMaterial.mNormalUVChannel == 0 ? inUV0 : inUV1 );

    vec3 N   = normalize( inNormal );
    vec3 T   = normalize( q1 * st2.t - q2 * st1.t );
    vec3 B   = -normalize( cross( N, T ) );
    mat3 TBN = mat3( T, B, N );

    // // Calculate normal in tangent space
    // vec3 N     = normalize( inNormal );
    // vec3 T     = normalize( inTangent );
    // vec3 B     = cross( N, T );
    // mat3 TBN   = mat3( T, B, N );
    vec3 tnorm = TBN * normalize( tangentNormal );
    outNormal  = vec4( tnorm, 1.0 );
    outAlbedo = texture( gTextures[lMaterial.mBaseColorTextureID], lMaterial.mBaseColorUVChannel == 0 ? inUV0 : inUV1 );
    outSpecular = vec4(0.0, 0.0, 0.0, 1.0);
}