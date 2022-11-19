#version 460
#extension GL_EXT_nonuniform_qualifier : enable

layout( location = 0 ) in vec3 inWorldPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;
layout( location = 3 ) in vec2 inUV1;

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

layout( location = 0 ) out vec4 outColor;

const float PI             = 3.14159265359;
const float M_PI           = 3.141592653589793;
const float c_MinRoughness = 0.04;

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

vec4 tonemap( vec4 color )
{
    vec3 outcol = Uncharted2Tonemap( color.rgb * uboParams.exposure );
    outcol      = outcol * ( 1.0f / Uncharted2Tonemap( vec3( 11.2f ) ) );
    return vec4( pow( outcol, vec3( 1.0f / uboParams.gamma ) ), color.a );
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

// ----------------------------------------------------------------------------
float DistributionGGX( vec3 N, vec3 H, float roughness )
{
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max( dot( N, H ), 0.0 );
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = ( NdotH2 * ( a2 - 1.0 ) + 1.0 );
    denom       = PI * denom * denom;

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
vec3 fresnelSchlick( float cosTheta, vec3 F0 ) { return F0 + ( 1.0 - F0 ) * pow( clamp( 1.0 - cosTheta, 0.0, 1.0 ), 5.0 ); }

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

// ----------------------------------------------------------------------------
void main()
{
    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    vec4 lBaseColor = lMaterial.mBaseColorFactor;
    lBaseColor =
        SRGBtoLINEAR( texture( gTextures[lMaterial.mBaseColorTextureID], lMaterial.mBaseColorUVChannel == 0 ? inUV0 : inUV1 ) ) *
        lMaterial.mBaseColorFactor;

    float metallic      = clamp( lMaterial.mMetallicFactor, 0.0, 1.0 );
    float roughness     = clamp( lMaterial.mRoughnessFactor, c_MinRoughness, 1.0 );
    vec4  lSampledValue = texture( gTextures[lMaterial.mMetalnessTextureID], lMaterial.mMetalnessUVChannel == 0 ? inUV0 : inUV1 );
    metallic            = lSampledValue.r * lMaterial.mMetallicFactor;
    roughness           = lSampledValue.g * lMaterial.mRoughnessFactor;

    vec3 emissive = vec3( lMaterial.mEmissiveFactor );
    emissive =
        SRGBtoLINEAR( texture( gTextures[lMaterial.mEmissiveTextureID], lMaterial.mEmissiveUVChannel == 0 ? inUV0 : inUV1 ) ).rgb *
        vec3( lMaterial.mEmissiveFactor );

    vec3 N;
    if( lMaterial.mNormalTextureID == 0 )
        N = normalize( inNormal );
    else
        N = getNormalFromMap( gTextures[lMaterial.mNormalTextureID], lMaterial.mNormalUVChannel == 0 ? inUV0 : inUV1 );

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
        vec3 lLightPosition = ubo.PointLights[lPointLightIndex].WorldPosition;
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

    float ao  = texture( gTextures[lMaterial.mOcclusionTextureID], ( lMaterial.mOcclusionUVChannel == 0 ? inUV0 : inUV1 ) ).r;
    hdr_color = mix( hdr_color, hdr_color * ao, lMaterial.mOcclusionStrength );

    hdr_color = hdr_color + emissive;
    // vec3 hdr_color = Lo;
    vec4 fullcolor = vec4( hdr_color, lBaseColor.a );

    outColor = tonemap( fullcolor );
    // outColor = vec4(vec3((length(inWorldPos) - 0.5)), 1.0);
}
