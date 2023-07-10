#extension GL_EXT_nonuniform_qualifier : enable

layout( location = 0 ) in vec3 inWorldPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;

layout( set = 2, binding = 0 ) uniform sampler2D gDirectionalShadowMaps[];
layout( set = 3, binding = 0 ) uniform sampler2D gSpotlightShadowMaps[];
layout( set = 4, binding = 0 ) uniform samplerCube gPointLightShadowMaps[];

#define MAX_NUM_LIGHTS 64


struct DirectionalLightData
{
    vec3  Direction;
    vec3  Color;
    float Intensity;
    mat4  Transform;
    int   IsOn;
};

struct PointLightData
{
    vec3  WorldPosition;
    vec3  Color;
    float Intensity;
    int   IsOn;
};

struct SpotlightData
{
    vec3  WorldPosition;
    vec3  LookAtDirection;
    vec3  Color;
    float Intensity;
    float Cone;
    mat4  Transform;
    int   IsOn;
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

const mat4 biasMat = mat4( 
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0 );

#define EPSILON 0.15

layout( location = 0 ) out vec4 outFragcolor;

const float PI             = 3.14159265359;
const float M_PI           = 3.141592653589793;
const float c_MinRoughness = 0.04;

float textureProj(sampler2D shadowMap, vec4 shadowCoord, vec2 off)
{
    float shadow = 1.0;
    if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
    {
        float dist = texture( shadowMap, shadowCoord.st + off ).r;
        if ( shadowCoord.w > 0.0 && dist < shadowCoord.z ) 
        {
            shadow = uboParams.AmbientLightIntensity;
        }
    }
    return shadow;
}

float filterPCF(sampler2D shadowMap, vec4 sc)
{
    ivec2 texDim = textureSize(shadowMap, 0);
    float scale = 1.5;
    float dx = scale * 1.0 / float(texDim.x);
    float dy = scale * 1.0 / float(texDim.y);

    float shadowFactor = 0.0;
    int count = 0;
    int range = 1;
    
    for (int x = -range; x <= range; x++)
    {
        for (int y = -range; y <= range; y++)
        {
            shadowFactor += textureProj(shadowMap, sc, vec2(dx*x, dy*y));
            count++;
        }
    
    }
    return shadowFactor / count;
}

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

