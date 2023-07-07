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