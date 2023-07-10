#extension GL_EXT_nonuniform_qualifier : enable

layout( location = 0 ) in vec2 inUV;

layout( set = 1, binding = 0 ) uniform sampler2D samplerPosition;
layout( set = 1, binding = 1 ) uniform sampler2D samplerNormal;
layout( set = 1, binding = 2 ) uniform sampler2D samplerAlbedo;
layout( set = 1, binding = 3 ) uniform sampler2D samplerOcclusionMetalRough;

layout( set = 2, binding = 0 ) uniform sampler2D gDirectionalShadowMaps[];
layout( set = 3, binding = 0 ) uniform sampler2D gSpotlightShadowMaps[];
layout( set = 4, binding = 0 ) uniform samplerCube gPointLightShadowMaps[];


layout( location = 0 ) out vec4 outFragcolor;

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

const float PI = 3.14159265359;

struct LightData
{
    // The color (.rgb) and pre-exposed intensity (.w) of the light. The color is an RGB value in the linear sRGB color space.
    // The pre-exposed intensity is the intensity of the light multiplied by the camera's exposure value.
    vec4  mColorIntensity;

    // The normalized light vector, in world space (direction from the current fragment's position to the light).
    vec3  mL;

    // The normalized light half vector, in world space (direction from the current fragment's position to the light).
    vec3  mH;

    // The dot product of the shading normal (with normal mapping applied) and the light vector. This value is equal to the result of
    // saturate(dot(getWorldSpaceNormal(), lightData.l)). This value is always between 0.0 and 1.0. When the value is <= 0.0,
    // the current fragment is not visible from the light and lighting computations can be skipped.
    float mNdotL;

    // The position of the light in world space.
    vec3  mWorldPosition;

    // Attenuation of the light based on the distance from the current fragment to the light in world space. This value between 0.0 and 1.0
    // is computed differently for each type of light (it's always 1.0 for directional lights).
    float mAttenuation;

    // Visibility factor computed from shadow maps or other occlusion data specific to the light being evaluated. This value is between 0.0 and
    // 1.0.
    float mVisibility;    
};

const int enablePCF = 1;


#include "Common/ToneMap.glsl"
#include "Common/PBRFunctions.glsl"


float TextureProj(sampler2D shadowMap, vec4 shadowCoord, vec2 off)
{
    float shadow = 1.0;
    if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
    {
        float dist = texture( shadowMap, shadowCoord.st + off ).r;
        if ( shadowCoord.w > 0.0 && dist < shadowCoord.z ) 
        {
            shadow = 0.0;
        }
    }
    return shadow;
}

float FilterPCF(sampler2D shadowMap, vec4 sc)
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
            shadowFactor += TextureProj(shadowMap, sc, vec2(dx*x, dy*y));
            count++;
        }
    
    }
    return shadowFactor / count;
}

const mat4 biasMat = mat4( 
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0 );

#define EPSILON 0.15

// float linearize_depth(float d,float zNear,float zFar)
// {
//     return zNear * zFar / (zFar + d * (zNear - zFar));
// }


void ComputeDirectionalLightData(vec3 inWorldPos, vec3 aSurfaceNormal, vec3 aEyeDirection, sampler2D aShadowMap, DirectionalLightData aInData, inout LightData aLightData)
{
    aLightData.mColorIntensity = vec4(aInData.Color, aInData.Intensity);
    aLightData.mL = normalize( aInData.Direction );
    aLightData.mH = normalize( aEyeDirection + aLightData.mL );
    aLightData.mNdotL = clamp( dot( aSurfaceNormal, aLightData.mL ), 0.0, 1.0 );
    aLightData.mWorldPosition = vec3(0.0);
    aLightData.mAttenuation = 1.0;

    vec4 lShadowNormalizedCoordinates = biasMat * aInData.Transform * vec4(inWorldPos, 1.0f);
    lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

    if (enablePCF == 1) 
        aLightData.mVisibility = FilterPCF(aShadowMap, lShadowNormalizedCoordinates);
    else 
        aLightData.mVisibility = TextureProj(aShadowMap, lShadowNormalizedCoordinates, vec2(0.0));
}

void ComputePointLightData(vec3 inWorldPos, vec3 aSurfaceNormal, vec3 aEyeDirection, samplerCube aShadowMap, PointLightData aInData, inout LightData aLightData)
{
    aLightData.mColorIntensity = vec4(aInData.Color, aInData.Intensity);
    aLightData.mL = normalize( aInData.WorldPosition - inWorldPos );
    aLightData.mH = normalize( aEyeDirection + aLightData.mL );
    aLightData.mNdotL = clamp( dot( aSurfaceNormal, aLightData.mL ), 0.0, 1.0 );
    aLightData.mWorldPosition = aInData.WorldPosition;
    aLightData.mVisibility = 1.0;

    vec3 v = aLightData.mWorldPosition - inWorldPos;
    float lDistanceSqared = dot(v,v);
    aLightData.mAttenuation = 1.0 / max(lDistanceSqared, 1e-4);

    vec3 coords = -aLightData.mL;
    float lShadowDistanceSquared = texture(aShadowMap, coords).r;
    aLightData.mVisibility = (lDistanceSqared <= lShadowDistanceSquared + EPSILON) ? 1.0 : 0.0;
}

void ComputeSpotLightData(vec3 inWorldPos, vec3 aSurfaceNormal, vec3 aEyeDirection, sampler2D aShadowMap, SpotlightData aInData, inout LightData aLightData)
{
    aLightData.mColorIntensity = vec4(aInData.Color, aInData.Intensity);
    aLightData.mL = -normalize( aInData.LookAtDirection );
    aLightData.mH = normalize( aEyeDirection + aLightData.mL );
    aLightData.mNdotL = clamp( dot( aSurfaceNormal, aLightData.mL ), 0.0, 1.0 );
    aLightData.mWorldPosition = aInData.WorldPosition;
    aLightData.mVisibility = 1.0;

    vec3 v = aLightData.mWorldPosition - inWorldPos;
    float lDistanceSqared = dot(v,v);
    aLightData.mAttenuation = 1.0 / max(lDistanceSqared, 1e-4);

    vec3  L = normalize( aInData.WorldPosition - inWorldPos );
    float lAngleToLightOrigin = clamp(dot( L, normalize( aLightData.mL ) ), 0.0, 1.0);

    if(lAngleToLightOrigin < aInData.Cone)
    {
        aLightData.mAttenuation *= 0.0;
    }

    vec4 lShadowNormalizedCoordinates = biasMat * aInData.Transform * vec4(inWorldPos, 1.0f);
    lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

    if (enablePCF == 1) 
        aLightData.mVisibility = FilterPCF(aShadowMap, lShadowNormalizedCoordinates);
    else 
        aLightData.mVisibility = TextureProj(aShadowMap, lShadowNormalizedCoordinates, vec2(0.0));
}

vec3 calculateLighting( vec3 inWorldPos, vec3 N, vec4 lBaseColor, vec4 aometalrough, vec4 emissive, ShadingData aShadingData )
{
    float metallic  = aometalrough.g;
    float roughness = aometalrough.b;

    vec3 V = normalize( ubo.camPos - inWorldPos );

    // reflectance equation
    vec3 Lo = vec3( 0.0f );

    for( int i = 0; i < ubo.DirectionalLightCount; i++ )
    {
        if (ubo.DirectionalLights[i].IsOn == 0) continue;
        
        LightData lLightData;
        ComputeDirectionalLightData(inWorldPos, N, V, gDirectionalShadowMaps[i], ubo.DirectionalLights[i], lLightData);
        Lo += ComputeLightContribution( N, V, aShadingData, lLightData );
    }

    for( int i = 0; i < ubo.PointLightCount; i++ )
    {
        if (ubo.PointLights[i].IsOn == 0) continue;

        LightData lLightData;
        ComputePointLightData(inWorldPos, N, V, gPointLightShadowMaps[i], ubo.PointLights[i], lLightData);
        Lo += ComputeLightContribution( N, V, aShadingData, lLightData );
    }

    for( int i = 0; i < ubo.SpotlightCount; i++ )
    {
        if (ubo.Spotlights[i].IsOn == 0) continue;

        LightData lLightData;
        ComputeSpotLightData(inWorldPos, N, V, gSpotlightShadowMaps[i], ubo.Spotlights[i], lLightData);
        Lo += ComputeLightContribution( N, V, aShadingData, lLightData );
    }

    return Lo ;
}


void main()
{
    vec3 fragColor = vec3( 0.0f );
    vec4 emissive  = vec4( 0.0f );

    vec3 pos    = texture( samplerPosition, inUV ).rgb;
    vec3 normal = texture( samplerNormal, inUV ).rgb;
    vec4 albedo = texture( samplerAlbedo, inUV );

    vec4 metalrough = texture( samplerOcclusionMetalRough, inUV );

    ShadingData lShadingData;
    ComputeShadingData(albedo.rgb * albedo.a, vec3(0.04), metalrough.g, metalrough.b, lShadingData);

    fragColor += calculateLighting( pos, normal, albedo, metalrough, emissive, lShadingData );
    float ao         = metalrough.r;
    float aoStrength = 0.0f;//metalrough.a;

    vec3 ambient   = uboParams.AmbientLightIntensity * uboParams.AmbientLightColor.rgb * albedo.xyz;
    vec3 hdr_color = fragColor + ambient;
    hdr_color      = mix( hdr_color, hdr_color * ao, aoStrength );
    hdr_color      = hdr_color + emissive.xyz;

    vec3 lTonemappedColor = tonemap( hdr_color );

    int lLumaAsAlpha = 0;
    float lLuma = Luminance( lTonemappedColor );
    float alpha = (lLumaAsAlpha == 0 ) ? 1.0 : lLuma;

    if( uboParams.grayscaleRendering == 1.0f )
    {
        vec3 outColor = pow( vec3(lLuma), vec3( 1.0f / uboParams.gamma ) );
        outFragcolor = vec4( outColor, alpha);
    }
    else
    {   
        vec3 outColor = pow( lTonemappedColor, vec3( 1.0f / uboParams.gamma ) );
        outFragcolor = vec4( outColor, alpha );
    }

    // outFragcolor = vec4( fragColor, 1.0 );
    // outFragcolor = texture( gDirectionalShadowMaps[0], inUV );
}