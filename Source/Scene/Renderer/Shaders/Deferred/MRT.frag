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
layout( location = 3 ) out vec4 outOcclusionMetalRough;

const float c_MinRoughness = 0.04;

vec3 getNormalFromMap( sampler2D aNormalSampler, vec2 aCoords )
{
    // Perturb normal, see http://www.thetenthplanet.de/archives/1180
    vec3 tangentNormal = normalize( texture( aNormalSampler, aCoords ).xyz * 2.0 - vec3( 1.0 ) );

    vec3 dp1  = dFdx( inWorldPos );
    vec3 dp2  = dFdy( inWorldPos );
    vec2 duv1 = dFdx( aCoords );
    vec2 duv2 = dFdy( aCoords );

    // solve the linear system
    vec3 dp1perp = cross( inNormal, dp1 );
    vec3 dp2perp = cross( dp2, inNormal );
    vec3 T       = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B       = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame
    float invmax = inversesqrt( max( dot( T, T ), dot( B, B ) ) );

    return normalize( mat3( T * invmax, B * invmax, inNormal ) * tangentNormal );
}

void main()
{
    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    outPosition = vec4( inWorldPos, 1.0 );

    vec3 tnorm;
    if( lMaterial.mNormalTextureID == 0 )
        tnorm = normalize( inNormal );
    else
        tnorm = getNormalFromMap( gTextures[lMaterial.mNormalTextureID], lMaterial.mNormalUVChannel == 0 ? inUV0 : inUV1 );

    vec4  lSampledValue = texture( gTextures[lMaterial.mMetalnessTextureID], lMaterial.mMetalnessUVChannel == 0 ? inUV0 : inUV1 );
    float metallic      = lSampledValue.r * clamp( lMaterial.mMetallicFactor, 0.0, 1.0 );
    float roughness     = lSampledValue.g * clamp( lMaterial.mRoughnessFactor, c_MinRoughness, 1.0 );
    float ao = texture( gTextures[lMaterial.mOcclusionTextureID], ( lMaterial.mOcclusionUVChannel == 0 ? inUV0 : inUV1 ) ).r *
               lMaterial.mOcclusionStrength;
    metallic  = lSampledValue.r * lMaterial.mMetallicFactor;
    roughness = lSampledValue.g * lMaterial.mRoughnessFactor;

    outNormal = vec4( tnorm, 1.0 );
    outAlbedo = texture( gTextures[lMaterial.mBaseColorTextureID], lMaterial.mBaseColorUVChannel == 0 ? inUV0 : inUV1 ) *
                lMaterial.mBaseColorFactor;
    // outAlbedo = outNormal;
    outOcclusionMetalRough = vec4( ao, metallic, roughness, lMaterial.mOcclusionStrength );
}