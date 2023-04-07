#version 460
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


#include "../Common/ToneMap.glsli"
#include "../Common/PBRFunctions.glsli"


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

const mat4 biasMat = mat4( 
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0 );

#define EPSILON 0.15

float linearize_depth(float d,float zNear,float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

vec3 calculateLighting( vec3 inWorldPos, vec3 N, vec4 lBaseColor, vec4 aometalrough, vec4 emissive )
{
    float metallic  = aometalrough.g;
    float roughness = aometalrough.b;

    vec3 V = normalize( ubo.camPos - inWorldPos );

    int enablePCF = 1;

    // reflectance equation
    vec3 Lo = vec3( 0.0f );

    for( int i = 0; i < ubo.DirectionalLightCount; i++ )
    {
        if (ubo.DirectionalLights[i].IsOn == 0) continue;
        
        vec3 radiance = ubo.DirectionalLights[i].Color * ubo.DirectionalLights[i].Intensity;
        vec3 lLightDirection = normalize( ubo.DirectionalLights[i].Direction );

        vec3 lLightContribution = ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, radiance, metallic, roughness );
        vec4 lShadowNormalizedCoordinates = biasMat * ubo.DirectionalLights[i].Transform * vec4(inWorldPos, 1.0f);
        lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

        float lShadowFactor = 1.0f;
        if (enablePCF == 1) 
            lShadowFactor = filterPCF(gDirectionalShadowMaps[i], lShadowNormalizedCoordinates);
        else 
            lShadowFactor = textureProj(gDirectionalShadowMaps[i], lShadowNormalizedCoordinates, vec2(0.0));

        Lo += (lLightContribution * lShadowFactor);
    }

    for( int i = 0; i < ubo.PointLightCount; i++ )
    {
        if (ubo.PointLights[i].IsOn == 0) continue;

        vec3 lLightPosition  = ubo.PointLights[i].WorldPosition;
        vec3 lLightDirection = normalize( lLightPosition - inWorldPos );

        vec3 lRadiance = ComputeRadiance( ubo.PointLights[i].WorldPosition, inWorldPos,
                                          ubo.PointLights[i].Color, ubo.PointLights[i].Intensity );

        vec3 lLightContribution = ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, lRadiance, metallic, roughness );

        float lShadowFactor = 1.0f;
        vec3 lightVec = inWorldPos - lLightPosition;
        vec3 lTexCoord = normalize(lightVec);

        float sampledDist = texture(gPointLightShadowMaps[i], lTexCoord).r;
        float dist = length(lightVec);
        lShadowFactor = (dist <= sampledDist + EPSILON) ? 1.0 : uboParams.AmbientLightIntensity;

        Lo += (lLightContribution * lShadowFactor);
    }

    for( int i = 0; i < ubo.SpotlightCount; i++ )
    {
        if (ubo.Spotlights[i].IsOn == 0) continue;

        vec3  L                   = normalize( ubo.Spotlights[i].WorldPosition - inWorldPos );
        vec3  lLightDirection     = normalize( ubo.Spotlights[i].LookAtDirection );
        float lAngleToLightOrigin = dot( L, normalize( -lLightDirection ) );

        if( lAngleToLightOrigin < ubo.Spotlights[i].Cone ) continue;

        vec3 lRadiance = ComputeRadiance( ubo.Spotlights[i].WorldPosition, inWorldPos,
                                          ubo.Spotlights[i].Color, ubo.Spotlights[i].Intensity );

        vec3 lLightContribution = ComputeLightContribution( lBaseColor.xyz, N, V, L, lRadiance, metallic, roughness );
        vec4 lShadowNormalizedCoordinates = biasMat * ubo.Spotlights[i].Transform * vec4(inWorldPos, 1.0f);
        lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

        float lShadowFactor = 1.0f;
        if (enablePCF == 1) 
            lShadowFactor = filterPCF(gSpotlightShadowMaps[i], lShadowNormalizedCoordinates);
        else 
            lShadowFactor = textureProj(gSpotlightShadowMaps[i], lShadowNormalizedCoordinates, vec2(0.0));

        Lo += (lLightContribution * lShadowFactor);
    }

    return Lo ;
}

vec4 SRGBtoLINEAR( vec4 srgbIn )
{
    vec3 bLess  = step( vec3( 0.04045 ), srgbIn.xyz );
    vec3 linOut = mix( srgbIn.xyz / vec3( 12.92 ), pow( ( srgbIn.xyz + vec3( 0.055 ) ) / vec3( 1.055 ), vec3( 2.4 ) ), bLess );
    return vec4( linOut, srgbIn.w );
}



void main()
{
    vec3 fragColor = vec3( 0.0f );
    vec4 emissive  = vec4( 0.0f );

    vec3 pos    = texture( samplerPosition, inUV ).rgb;
    vec3 normal = texture( samplerNormal, inUV ).rgb;
    vec4 albedo = SRGBtoLINEAR( texture( samplerAlbedo, inUV ) );

    vec4 metalrough = texture( samplerOcclusionMetalRough, inUV );

    fragColor += calculateLighting( pos, normal, albedo, metalrough, emissive );
    float ao         = metalrough.r;
    float aoStrength = metalrough.a;

    vec3 ambient   = uboParams.AmbientLightIntensity * uboParams.AmbientLightColor.rgb * albedo.xyz;
    vec3 hdr_color = fragColor + ambient;
    hdr_color      = mix( hdr_color, hdr_color * ao, aoStrength );
    hdr_color      = hdr_color + emissive.xyz;

    vec3 outColor = tonemap( hdr_color );
    outColor = pow( outColor, vec3( 1.0f / uboParams.gamma ) );

    // float lLuma = dot( outColor, vec3( 0.2126, 0.7152, 0.0722 ) );
    float lLuma = dot( outColor, vec3( 0.299, 0.587, 0.114 ) );

    int lLumaAsAlpha = 0;

    if( uboParams.grayscaleRendering == 1.0f )
        outFragcolor = vec4( vec3(lLuma), (lLumaAsAlpha == 0 ) ? 1.0 : lLuma);
    else
        outFragcolor = vec4( outColor, (lLumaAsAlpha == 0 ) ? 1.0 : lLuma );

    // outFragcolor = vec4( normal, 1.0 );
    // outFragcolor = texture( gDirectionalShadowMaps[0], inUV );
}