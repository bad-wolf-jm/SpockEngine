#version 460

layout( location = 0 ) in vec2 inUV;

layout( set = 1, binding = 0 ) uniform sampler2DMS samplerPosition;
layout( set = 1, binding = 1 ) uniform sampler2DMS samplerNormal;
layout( set = 1, binding = 2 ) uniform sampler2DMS samplerAlbedo;
layout( set = 1, binding = 3 ) uniform sampler2DMS samplerOcclusionMetalRough;

layout( location = 0 ) out vec4 outFragcolor;

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

layout( constant_id = 0 ) const int NUM_SAMPLES = 4;

#define NUM_LIGHTS 6
const float PI = 3.14159265359;

// Manual resolve for MSAA samples
vec4 resolve( sampler2DMS tex, ivec2 uv )
{
    vec4 result = vec4( 0.0 );
    for( int i = 0; i < NUM_SAMPLES; i++ )
    {
        vec4 val = texelFetch( tex, uv, i );
        result += val;
    }
    // Average resolved samples
    return result / float( NUM_SAMPLES );
}

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
vec3 fresnelSchlick( float cosTheta, vec3 F0 ) 
{ 
    return F0 + ( 1.0 - F0 ) * pow( clamp( 1.0 - cosTheta, 0.0, 1.0 ), 5.0 ); 
}

vec3 CookTorrance( vec3 F0, vec3 N, vec3 L, vec3 V, vec3 H, float roughness )
{
    // Cook-Torrance BRDF
    float NDF = DistributionGGX( N, H, roughness );
    float G   = GeometrySmith( N, V, L, roughness );
    vec3  F   = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );
    return ( NDF * G * F ) / ( 4 * max( dot( N, V ), 0.0 ) * max( dot( N, L ), 0.0 ) + 0.0001 );
}

vec3 calculateLighting( vec3 inWorldPos, vec3 N, vec4 lBaseColor, vec4 aometalrough, vec4 emissive )
{
    float metallic  = aometalrough.g;
    float roughness = aometalrough.b;

    vec3 V  = normalize( ubo.camPos - inWorldPos );
    vec3 F0 = vec3( 0.04 );
    F0      = mix( F0, lBaseColor.xyz, metallic );

    // reflectance equation
    vec3 Lo = vec3( 0.0f );

    for( int l_DirectionalLightIndex = 0; l_DirectionalLightIndex < ubo.DirectionalLightCount; l_DirectionalLightIndex++ )
    {
        vec3 L = normalize( ubo.DirectionalLights[l_DirectionalLightIndex].Direction );
        vec3 H = normalize( V + L );
        vec3 radiance =
            ubo.DirectionalLights[l_DirectionalLightIndex].Color * ubo.DirectionalLights[l_DirectionalLightIndex].Intensity;

        // Cook-Torrance BRDF
        vec3 specular = CookTorrance( F0, N, L, V, H, roughness );

        // kS is equal to Fresnel
        vec3 kS = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );

        // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
        // this relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3( 1.0 ) - kS;

        // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal
        // (pure metals have no diffuse light).
        kD *= ( 1.0 - metallic );

        // scale light by NdotL
        float NdotL = max( dot( N, L ), 0.0 );

        // add to outgoing radiance Lo
        Lo += ( kD * lBaseColor.xyz / PI + specular ) * radiance * NdotL;
        // Lo += kD;
    }

    for( int l_PointLightIndex = 0; l_PointLightIndex < ubo.PointLightCount; l_PointLightIndex++ )
    {
        vec3 l_LightPosition = ubo.PointLights[l_PointLightIndex].WorldPosition;

        vec3  L           = normalize( l_LightPosition - inWorldPos );
        vec3  H           = normalize( V + L );
        float distance    = length( l_LightPosition - inWorldPos );
        float attenuation = 1.0 / ( distance * distance );
        vec3  radiance    = ubo.PointLights[l_PointLightIndex].Color * ubo.PointLights[l_PointLightIndex].Intensity * attenuation;

        // Cook-Torrance BRDF
        vec3 specular = CookTorrance( F0, N, L, V, H, roughness );

        // kS is equal to Fresnel
        vec3 kS = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );

        // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
        // this relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3( 1.0 ) - kS;

        // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal
        // (pure metals have no diffuse light).
        kD *= ( 1.0 - metallic );

        // scale light by NdotL
        float NdotL = clamp( dot( N, L ), 0.000001, 1.0 );

        // add to outgoing radiance Lo
        Lo += ( kD * lBaseColor.xyz / PI + specular ) * radiance * NdotL;
        // Lo += ( lBaseColor.xyz / PI + specular ) * radiance * NdotL;
        // Lo += normalize(inWorldPos);
    }

    for( int l_SpotlightIndex = 0; l_SpotlightIndex < ubo.SpotlightCount; l_SpotlightIndex++ )
    {
        vec3  L                    = normalize( ubo.Spotlights[l_SpotlightIndex].WorldPosition - inWorldPos );
        vec3  l_LightDirection     = normalize( ubo.Spotlights[l_SpotlightIndex].LookAtDirection );
        float l_AngleToLightOrigin = dot( L, normalize( -l_LightDirection ) );

        if( l_AngleToLightOrigin < ubo.Spotlights[l_SpotlightIndex].Cone ) continue;

        vec3  H           = normalize( V + L );
        float distance    = length( ubo.Spotlights[l_SpotlightIndex].WorldPosition - inWorldPos );
        float attenuation = 1.0 / ( distance * distance );
        vec3  radiance    = ubo.Spotlights[l_SpotlightIndex].Color * ubo.Spotlights[l_SpotlightIndex].Intensity * attenuation;

        // Cook-Torrance BRDF
        vec3 specular = CookTorrance( F0, N, L, V, H, roughness );

        // kS is equal to Fresnel
        vec3 kS = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );

        // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve
        // this relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3( 1.0 ) - kS;

        // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal
        // (pure metals have no diffuse light).
        kD *= ( 1.0 - metallic );

        // scale light by NdotL
        float NdotL = max( dot( N, L ), 0.0 );

        // add to outgoing radiance Lo
        Lo += ( kD * lBaseColor.xyz / PI + specular ) * radiance * NdotL;
    }

    return Lo;
}

void main()
{
    ivec2 attDim = textureSize( samplerPosition );
    ivec2 UV     = ivec2( inUV * attDim );

    // Ambient part
    vec4 alb       = resolve( samplerAlbedo, UV );
    vec3 fragColor = vec3( 0.0f );
    vec4 emissive  = vec4( 0.0f );

    // Calualte lighting for every MSAA sample
    float ao = 0.0f;
    float aoStrength = 0.0f;
    for( int i = 0; i < NUM_SAMPLES; i++ )
    {
        vec3 pos        = texelFetch( samplerPosition, UV, i ).rgb;
        vec3 normal     = texelFetch( samplerNormal, UV, i ).rgb;
        vec4 albedo     = texelFetch( samplerAlbedo, UV, i );
        vec4 metalrough = texelFetch( samplerOcclusionMetalRough, UV, i );

        fragColor += calculateLighting( pos, normal, albedo, metalrough, emissive );
        ao += metalrough.r;
        aoStrength += metalrough.a;
    }

    vec3 ambient   = uboParams.AmbientLightIntensity * uboParams.AmbientLightColor.rgb * alb.xyz;
    vec3 hdr_color = ambient + fragColor / float( NUM_SAMPLES );
    hdr_color = mix( hdr_color, hdr_color * ao / float( NUM_SAMPLES ), aoStrength / float( NUM_SAMPLES ) );
    hdr_color = hdr_color + emissive.xyz;
    // fragColor = ( alb.rgb * ambient ) + fragColor / float( NUM_SAMPLES );
    // vec4 full_color = vec4( hdr_color, lBaseColor.a );

    vec3 outColor = tonemap( hdr_color );

    outFragcolor = vec4( outColor, 1.0 );
}