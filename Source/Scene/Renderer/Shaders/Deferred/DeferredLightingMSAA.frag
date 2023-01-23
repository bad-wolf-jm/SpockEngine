#version 460
#extension GL_EXT_nonuniform_qualifier : enable

layout( location = 0 ) in vec2 inUV;

layout( set = 1, binding = 0 ) uniform sampler2D samplerPosition;
layout( set = 1, binding = 1 ) uniform sampler2D samplerNormal;
layout( set = 1, binding = 2 ) uniform sampler2D samplerAlbedo;
layout( set = 1, binding = 3 ) uniform sampler2D samplerOcclusionMetalRough;

layout( set = 2, binding = 0 ) uniform sampler2D gDirectionalShadowMaps[];
layout( set = 3, binding = 0 ) uniform sampler2D gSpotlightShadowMaps[];


layout( location = 0 ) out vec4 outFragcolor;

#define MAX_NUM_LIGHTS 64

struct DirectionalLightData
{
    vec3  Direction;
    vec3  Color;
    float Intensity;
    mat4  Transform;
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
    mat4  Transform;
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

vec3 Uncharted2Tonemap( vec3 color )
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    float W = 11.2;
    return ( ( color * ( A * color + C * B ) + D * E ) / ( color * ( A * color + B ) + D * F ) ) - E / F;
}

vec3 tonemap( vec3 color )
{
    vec3 outcol = Uncharted2Tonemap( color.rgb * uboParams.exposure );
    outcol      = outcol * ( 1.0f / Uncharted2Tonemap( vec3( 11.2f ) ) );
    return pow( outcol, vec3( 1.0f / uboParams.gamma ) );
}

// ----------------------------------------------------------------------------
float DistributionGGX( vec3 N, vec3 H, float roughness )
{
    float a      = roughness * roughness;
    float a2     = a * a;
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
vec3 fresnelSchlick( float cosTheta, vec3 F0 ) { return F0 + ( vec3( 1.0 ) - F0 ) * pow( clamp( 1.0 - cosTheta, 0.0, 1.0 ), 5.0 ); }

vec3 CookTorrance( vec3 F0, vec3 N, vec3 L, vec3 V, vec3 H, float roughness )
{
    // Cook-Torrance BRDF
    float NDF = DistributionGGX( N, H, roughness );
    float G   = GeometrySmith( N, V, L, roughness );
    vec3  F   = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );
    return ( NDF * G * F ) / ( 4 * max( dot( N, V ), 0.0 ) * max( dot( N, L ), 0.0 ) + 0.0001 );
}

vec3 ComputeLightContribution( vec3 aBaseColor, vec3 aSurfaceNormal, vec3 aEyeDirection, vec3 aLightDirection, vec3 aRadiance,
                               float aMetal, float aRough )
{

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the base color as F0 (metallic workflow)
    vec3 lF0 = mix( vec3( 0.04 ), aBaseColor, aMetal );

    vec3 H = normalize( aEyeDirection + aLightDirection );

    // Cook-Torrance BRDF
    vec3 lSpecular = CookTorrance( lF0, aSurfaceNormal, aLightDirection, aEyeDirection, H, aRough );

    // kS is equal to Fresnel
    vec3 kS = fresnelSchlick( max( dot( H, aEyeDirection ), 0.0 ), lF0 );

    // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
    // this relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3( 1.0 ) - kS;

    // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal
    // (pure metals have no diffuse light).
    kD *= ( 1.0 - aMetal );

    // scale light by NdotL
    float NdotL = max( dot( aSurfaceNormal, aLightDirection ), 0.0 );

    // add to outgoing radiance Lo
    return ( kD * aBaseColor / PI + lSpecular ) * aRadiance * NdotL;
}

vec3 ComputeRadiance( vec3 aLightPosition, vec3 aObjectPosition, vec3 aLightColor, float aLightIntensity )
{
    float lDistance    = length( aLightPosition - aObjectPosition );
    float lAttenuation = 1.0 / ( lDistance * lDistance );

    return aLightColor * aLightIntensity * lAttenuation;
}

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
        vec3 radiance = ubo.DirectionalLights[i].Color * ubo.DirectionalLights[i].Intensity;
        vec3 lLightDirection = normalize( ubo.DirectionalLights[i].Direction );

        vec3 lLightContribution = ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, radiance, metallic, roughness );
        vec4 lShadowNormalizedCoordinates = biasMat * ubo.DirectionalLights[i].Transform * vec4(inWorldPos, 1.0f);
        lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

        float lShadow = (enablePCF == 1) ? filterPCF(gDirectionalShadowMaps[i], lShadowNormalizedCoordinates) : textureProj(gDirectionalShadowMaps[i], lShadowNormalizedCoordinates, vec2(0.0));

        lLightContribution *= lShadow;

        Lo += lLightContribution;
    }

    for( int i = 0; i < ubo.PointLightCount; i++ )
    {
        vec3 lLightPosition  = ubo.PointLights[i].WorldPosition;
        vec3 lLightDirection = normalize( lLightPosition - inWorldPos );

        vec3 lRadiance = ComputeRadiance( ubo.PointLights[i].WorldPosition, inWorldPos,
                                          ubo.PointLights[i].Color, ubo.PointLights[i].Intensity );
        Lo += ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, lRadiance, metallic, roughness );
    }

    for( int i = 0; i < ubo.SpotlightCount; i++ )
    {
        vec3  L                   = normalize( ubo.Spotlights[i].WorldPosition - inWorldPos );
        vec3  lLightDirection     = normalize( ubo.Spotlights[i].LookAtDirection );
        float lAngleToLightOrigin = dot( L, normalize( -lLightDirection ) );

        if( lAngleToLightOrigin < ubo.Spotlights[i].Cone ) continue;

        vec3 lRadiance = ComputeRadiance( ubo.Spotlights[i].WorldPosition, inWorldPos,
                                          ubo.Spotlights[i].Color, ubo.Spotlights[i].Intensity );

        vec3 lLightContribution = ComputeLightContribution( lBaseColor.xyz, N, V, L, lRadiance, metallic, roughness );

        vec4 lShadowNormalizedCoordinates = biasMat * ubo.Spotlights[i].Transform * vec4(inWorldPos, 1.0f);
        lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

        float lShadow = (enablePCF == 1) ? filterPCF(gSpotlightShadowMaps[i], lShadowNormalizedCoordinates) : textureProj(gSpotlightShadowMaps[i], lShadowNormalizedCoordinates, vec2(0.0));


        Lo += lLightContribution * lShadow;
    }

    return Lo;
}

vec4 SRGBtoLINEAR( vec4 srgbIn )
{
    vec3 bLess  = step( vec3( 0.04045 ), srgbIn.xyz );
    vec3 linOut = mix( srgbIn.xyz / vec3( 12.92 ), pow( ( srgbIn.xyz + vec3( 0.055 ) ) / vec3( 1.055 ), vec3( 2.4 ) ), bLess );
    return vec4( linOut, srgbIn.w );
}


vec3 shadow(vec3 fragcolor, vec3 inWorldPos) 
{
    int enablePCF = 1;

    float lShadowCoefficient = 1.0f;

    for( int i = 0; i < ubo.DirectionalLightCount; i++ )
    {
        vec4 lShadowNormalizedCoordinates = biasMat * ubo.DirectionalLights[i].Transform * vec4(inWorldPos, 1.0f);
        lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

        float lShadow = (enablePCF == 1) ? filterPCF(gDirectionalShadowMaps[i], lShadowNormalizedCoordinates) : textureProj(gDirectionalShadowMaps[i], lShadowNormalizedCoordinates, vec2(0.0));

        lShadowCoefficient *= lShadow;
    }

    for( int i = 0; i < ubo.SpotlightCount; i++ )
    {
        vec4 lShadowNormalizedCoordinates = biasMat * ubo.Spotlights[i].Transform * vec4(inWorldPos, 1.0f);
        lShadowNormalizedCoordinates /= lShadowNormalizedCoordinates.w;

        float lShadow = (enablePCF == 1) ? filterPCF(gSpotlightShadowMaps[i], lShadowNormalizedCoordinates) : textureProj(gSpotlightShadowMaps[i], lShadowNormalizedCoordinates, vec2(0.0));

        lShadowCoefficient *= lShadow;
    }

    return fragcolor * lShadowCoefficient;
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
    vec3 hdr_color = ambient + fragColor;
    hdr_color      = mix( hdr_color, hdr_color * ao, aoStrength );
    hdr_color      = hdr_color + emissive.xyz;

    // hdr_color = shadow(hdr_color, pos);

    vec3 outColor = tonemap( hdr_color );

    if( uboParams.grayscaleRendering == 1.0f )
        outFragcolor = vec4( vec3( dot( outColor, vec3( 0.2126, 0.7152, 0.0722 ) ) ), 1.0f );
    else
        outFragcolor = vec4( outColor, 1.0 );

    // outFragcolor = vec4( normal, 1.0 );
    // outFragcolor = texture( gDirectionalShadowMaps[0], inUV );
}