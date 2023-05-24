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

layout( push_constant ) uniform MaterialID { uint mMaterialID; }
material;

layout( location = 0 ) out vec4 outPosition;
layout( location = 1 ) out vec4 outNormal;
layout( location = 2 ) out vec4 outAlbedo;
layout( location = 3 ) out vec4 outOcclusionMetalRough;
layout( location = 4 ) out float outObjectID;

const float c_MinRoughness = 0.04;

#include "../Common/GetNormalFromMap.glsli"

struct MaterialInputs
{
    vec3  mNormal;
    vec4  mBaseColor;
    float mIsMetal;
    float mRoughness;
    float mOcclusionStrength;
    float mAmbiantOcclusion;
};

void InitializeMaterial(inout MaterialInputs aMaterial)
{
    aMaterial.mNormal            = vec3(0);
    aMaterial.mBaseColor         = vec4(0);
    aMaterial.mIsMetal           = 0.0;
    aMaterial.mRoughness         = 0.0;
    aMaterial.mOcclusionStrength = 0.0;
    aMaterial.mAmbiantOcclusion  = 0.0;
}

vec4 SRGBtoLINEAR( vec4 srgbIn )
{
    vec3 bLess  = step( vec3( 0.04045 ), srgbIn.xyz );
    vec3 linOut = mix( srgbIn.xyz / vec3( 12.92 ), pow( ( srgbIn.xyz + vec3( 0.055 ) ) / vec3( 1.055 ), vec3( 2.4 ) ), bLess );
    return vec4( linOut, srgbIn.w );
}


void Material(inout MaterialInputs aMaterial)
{
    InitializeMaterial(aMaterial);

    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    if( lMaterial.mNormalTextureID == 0 )
        aMaterial.mNormal = normalize( inNormal );
    else
        aMaterial.mNormal = getNormalFromMap( gTextures[lMaterial.mNormalTextureID], inUV0 );

    aMaterial.mBaseColor = SRGBtoLINEAR(texture( gTextures[lMaterial.mBaseColorTextureID], inUV0)) * lMaterial.mBaseColorFactor;

    vec4  lSampledValue  = texture( gTextures[lMaterial.mMetalnessTextureID], inUV0 );
    aMaterial.mIsMetal   = lSampledValue.r * clamp( lMaterial.mMetallicFactor, 0.0, 1.0 );
    aMaterial.mRoughness = lSampledValue.g * clamp( lMaterial.mRoughnessFactor, c_MinRoughness, 1.0 );

    aMaterial.mOcclusionStrength = lMaterial.mOcclusionStrength;
    aMaterial.mAmbiantOcclusion  = texture( gTextures[lMaterial.mOcclusionTextureID], inUV0 ).r * lMaterial.mOcclusionStrength;
}

void main()
{
    MaterialInputs lMaterialData;
    Material(lMaterialData);

    outPosition = vec4( inWorldPos, 1.0 );
    outNormal = vec4( lMaterialData.mNormal, 1.0 );
    outAlbedo = lMaterialData.mBaseColor;
    outOcclusionMetalRough = vec4( lMaterialData.mAmbiantOcclusion, lMaterialData.mIsMetal, lMaterialData.mRoughness, lMaterialData.mOcclusionStrength );
    outObjectID = 0.0f;
}