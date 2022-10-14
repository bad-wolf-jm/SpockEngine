#version 450

layout( location = 0 ) in vec3 inWorldPos;
layout( location = 1 ) in vec3 inNormal;
layout( location = 2 ) in vec2 inUV0;
layout( location = 3 ) in vec2 inUV1;

#define MAX_NUM_LIGHTS 64

struct DirectionalLightData
{
    vec3 Direction;
    vec3 Color;
    float Intensity;
};

struct PointLightData
{
    vec3 WorldPosition;
    vec3 Color;
    float Intensity;
};

struct SpotlightData
{
    vec3 WorldPosition;
    vec3 LookAtDirection;
    vec3 Color;
    float Intensity;
    float Cone;
};

// Scene bindings
layout( set = 0, binding = 0 ) uniform UBO
{
    mat4 projection;
    mat4 model;
    mat4 view;
    vec3 camPos;

    int DirectionalLightCount;
    DirectionalLightData DirectionalLights[MAX_NUM_LIGHTS];

    int SpotlightCount;
    SpotlightData Spotlights[MAX_NUM_LIGHTS];

    int PointLightCount;
    PointLightData PointLights[MAX_NUM_LIGHTS];
}
ubo;

layout( set = 0, binding = 1 ) uniform UBOParams
{
    float exposure;
    float gamma;
    float AmbientLightIntensity;
    vec4 AmbientLightColor;
    float debugViewInputs;
    float debugViewEquation;
}
uboParams;

// Material bindings
layout( set = 1, binding = 0 ) uniform sampler2D colorMap;
layout( set = 1, binding = 1 ) uniform sampler2D normalMap;
layout( set = 1, binding = 2 ) uniform sampler2D aoMap;
layout( set = 1, binding = 3 ) uniform sampler2D emissiveMap;
layout( set = 1, binding = 4 ) uniform sampler2D metalnessMap;
// layout( set = 1, binding = 5 ) uniform sampler2D roughnessMap;

layout( push_constant ) uniform Material
{
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    int baseColorTextureSet;
    int metalnessTextureSet;
    int roughnessTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float metallicFactor;
    float roughnessFactor;
    float alphaMask;
    float alphaMaskCutoff;
}
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
    ;
}

vec3 getNormalFromMap()
{
    // Perturb normal, see http://www.thetenthplanet.de/archives/1180
    if( material.normalTextureSet == -1.0 )
        return normalize( inNormal );

    vec3 tangentNormal = texture( normalMap, material.normalTextureSet == 0 ? inUV0 : inUV1 ).xyz * 2.0 - 1.0;

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
    vec3 F    = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );
    return ( NDF * G * F ) / ( 4 * max( dot( N, V ), 0.0 ) * max( dot( N, L ), 0.0 ) + 0.0001 );
}

// ----------------------------------------------------------------------------
void main()
{
    vec4 albedo = vec4( 0.0f );
    if( material.baseColorTextureSet > -1 )
    {
        albedo = SRGBtoLINEAR( texture( colorMap, material.baseColorTextureSet == 0 ? inUV0 : inUV1 ) ) * material.baseColorFactor;
    }
    else
    {
        albedo = material.baseColorFactor;
    }

    if( material.alphaMask == 1.0f )
    {
        if( albedo.a < material.alphaMaskCutoff )
        {
            discard;
        }
    }

    float metallic = material.metallicFactor;
    if( material.metalnessTextureSet > -1 )
    {
        metallic = texture( metalnessMap, material.metalnessTextureSet == 0 ? inUV0 : inUV1 ).r * metallic;
    }
    else
    {
        metallic = clamp( metallic, 0.0, 1.0 );
    }

    float roughness = material.roughnessFactor;
    if( material.roughnessTextureSet > -1 )
    {
        roughness = texture( metalnessMap, material.roughnessTextureSet == 0 ? inUV0 : inUV1 ).g * roughness;
    }
    else
    {
        roughness = clamp( roughness, c_MinRoughness, 1.0 );
    }

    const float u_OcclusionStrength = 1.0f;
    float ao                        = u_OcclusionStrength;
    if( material.occlusionTextureSet > -1 )
    {
        ao = texture( aoMap, ( material.occlusionTextureSet == 0 ? inUV0 : inUV1 ) ).r;
    }

    const float u_EmissiveFactor = 0.0f;
    vec3 emissive                = vec3( 0.0f );
    if( material.emissiveTextureSet > -1 )
    {
        emissive = SRGBtoLINEAR( texture( emissiveMap, material.emissiveTextureSet == 0 ? inUV0 : inUV1 ) ).rgb * u_EmissiveFactor;
    }

    vec3 N = getNormalFromMap();
    vec3 V = normalize( ubo.camPos - inWorldPos );

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
    vec3 F0 = vec3( 0.04 );
    F0      = mix( F0, albedo.xyz, metallic );

    // // reflectance equation
    vec3 Lo = vec3( 0.0f );

    for( int l_DirectionalLightIndex = 0; l_DirectionalLightIndex < ubo.DirectionalLightCount; l_DirectionalLightIndex++ )
    {
        vec3 L        = normalize( ubo.DirectionalLights[l_DirectionalLightIndex].Direction );
        vec3 H        = normalize( V + L );
        vec3 radiance = ubo.DirectionalLights[l_DirectionalLightIndex].Color * ubo.DirectionalLights[l_DirectionalLightIndex].Intensity;

        // Cook-Torrance BRDF
        vec3 specular = CookTorrance( F0, N, L, V, H, roughness );

        // kS is equal to Fresnel
        vec3 kS = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );

        // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3( 1.0 ) - kS;

        // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= ( 1.0 - metallic );

        // scale light by NdotL
        float NdotL = max( dot( N, L ), 0.0 );

        // add to outgoing radiance Lo
        Lo += ( kD * albedo.xyz / PI + specular ) * radiance * NdotL;
    }

    for( int l_PointLightIndex = 0; l_PointLightIndex < ubo.PointLightCount; l_PointLightIndex++ )
    {
        vec3 l_LightPosition = ubo.PointLights[l_PointLightIndex].WorldPosition;

        vec3 L            = normalize( l_LightPosition - inWorldPos );
        vec3 H            = normalize( V + L );
        float distance    = length( l_LightPosition - inWorldPos );
        float attenuation = 1.0 / ( distance * distance );
        vec3 radiance     = ubo.PointLights[l_PointLightIndex].Color * ubo.PointLights[l_PointLightIndex].Intensity * attenuation;

        // Cook-Torrance BRDF
        vec3 specular = CookTorrance( F0, N, L, V, H, roughness );

        // kS is equal to Fresnel
        vec3 kS = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );

        // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3( 1.0 ) - kS;

        // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= ( 1.0 - metallic );

        // scale light by NdotL
        float NdotL = clamp( dot( N, L ), 0.000001, 1.0 );

        // add to outgoing radiance Lo
        Lo += ( kD * albedo.xyz / PI + specular ) * radiance * NdotL;
        // Lo += NdotL;
    }

    for( int l_SpotlightIndex = 0; l_SpotlightIndex < ubo.SpotlightCount; l_SpotlightIndex++ )
    {
        vec3 L                     = normalize( ubo.Spotlights[l_SpotlightIndex].WorldPosition - inWorldPos );
        vec3 l_LightDirection      = normalize( ubo.Spotlights[l_SpotlightIndex].LookAtDirection );
        float l_AngleToLightOrigin = dot( L, normalize( -l_LightDirection ) );

        if( l_AngleToLightOrigin < ubo.Spotlights[l_SpotlightIndex].Cone )
            continue;

        vec3 H            = normalize( V + L );
        float distance    = length( ubo.Spotlights[l_SpotlightIndex].WorldPosition - inWorldPos );
        float attenuation = 1.0 / ( distance * distance );
        vec3 radiance     = ubo.Spotlights[l_SpotlightIndex].Color * ubo.Spotlights[l_SpotlightIndex].Intensity * attenuation;

        // Cook-Torrance BRDF
        vec3 specular = CookTorrance( F0, N, L, V, H, roughness );

        // kS is equal to Fresnel
        vec3 kS = fresnelSchlick( max( dot( H, V ), 0.0 ), F0 );

        // for energy conservation, the diffuse and specular light can't be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3( 1.0 ) - kS;

        // multiply kD by the inverse metalness such that only non-metals have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= ( 1.0 - metallic );

        // scale light by NdotL
        float NdotL = max( dot( N, L ), 0.0 );

        // add to outgoing radiance Lo
        Lo += ( kD * albedo.xyz / PI + specular ) * radiance * NdotL;
    }

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient = uboParams.AmbientLightIntensity * uboParams.AmbientLightColor.rgb * albedo.xyz * ao;

    vec3 hdr_color = ambient + Lo + emissive;
    // vec3 hdr_color = Lo;
    vec4 full_color = vec4( hdr_color, albedo.a );

    outColor = tonemap( full_color );
    // outColor = vec4(vec3((length(inWorldPos) - 0.5)), 1.0);
}
