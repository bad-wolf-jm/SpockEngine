void main()
{
    sShaderMaterial lMaterial = gMaterials.mArray[material.mMaterialID];

    vec4 lBaseColor = lMaterial.mBaseColorFactor;
    lBaseColor =
        SRGBtoLINEAR( texture( gTextures[lMaterial.mBaseColorTextureID], inUV0 ) ) *
        lMaterial.mBaseColorFactor;

    float metallic      = clamp( lMaterial.mMetallicFactor, 0.0, 1.0 );
    float roughness     = clamp( lMaterial.mRoughnessFactor, c_MinRoughness, 1.0 );
    vec4  lSampledValue = texture( gTextures[lMaterial.mMetalnessTextureID], inUV0 );
    metallic            = lSampledValue.r * lMaterial.mMetallicFactor;
    roughness           = lSampledValue.g * lMaterial.mRoughnessFactor;

    vec3 emissive = vec3( lMaterial.mEmissiveFactor );
    emissive =
        SRGBtoLINEAR( texture( gTextures[lMaterial.mEmissiveTextureID], inUV0 ) ).rgb *
        vec3( lMaterial.mEmissiveFactor );

    vec3 N;
    if( lMaterial.mNormalTextureID == 0 )
        N = normalize( inNormal );
    else
        N = getNormalFromMap( gTextures[lMaterial.mNormalTextureID], inUV0 );

    vec3 V = normalize( ubo.camPos - inWorldPos );

    // reflectance equation
    vec3 Lo = vec3( 0.0f );
    int enablePCF = 1;

    for( int i = 0; i < ubo.DirectionalLightCount; i++ )
    {
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
        // Lo += ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, radiance, metallic, roughness );
    }

    for( int i = 0; i < ubo.PointLightCount; i++ )
    {
        vec3 lLightPosition  = ubo.PointLights[i].WorldPosition;
        vec3 lLightDirection = normalize( lLightPosition - inWorldPos );

        vec3 lRadiance = ComputeRadiance( ubo.PointLights[i].WorldPosition, inWorldPos,
                                          ubo.PointLights[i].Color, ubo.PointLights[i].Intensity );
        // Lo += ComputeLightContribution( lBaseColor.xyz, N, V, lLightDirection, lRadiance, metallic, roughness );
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
        vec3  L                   = normalize( ubo.Spotlights[i].WorldPosition - inWorldPos );
        vec3  lLightDirection     = normalize( ubo.Spotlights[i].LookAtDirection );
        float lAngleToLightOrigin = dot( L, normalize( -lLightDirection ) );

        if( lAngleToLightOrigin < ubo.Spotlights[i].Cone ) continue;

        vec3 lRadiance = ComputeRadiance( ubo.Spotlights[i].WorldPosition, inWorldPos,
                                          ubo.Spotlights[i].Color, ubo.Spotlights[i].Intensity );

        // Lo += ComputeLightContribution( lBaseColor.xyz, N, V, L, lRadiance, metallic, roughness );

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

    // ambient lighting (note that the next IBL tutorial will replace
    // this ambient lighting with environment lighting).
    vec3 ambient   = uboParams.AmbientLightIntensity * uboParams.AmbientLightColor.rgb * lBaseColor.xyz;
    vec3 hdr_color = ambient + Lo;

    float ao  = texture( gTextures[lMaterial.mOcclusionTextureID], inUV0 ).r;
    hdr_color = mix( hdr_color, hdr_color * ao, lMaterial.mOcclusionStrength );

    hdr_color = hdr_color + emissive;
    // vec3 hdr_color = Lo;
    // vec4 fullcolor = vec4( hdr_color, lBaseColor.a );

    vec3 lOutColor = tonemap( hdr_color );
    lOutColor = pow( lOutColor, vec3( 1.0f / uboParams.gamma ) );

    float lLuma = dot( lOutColor, vec3( 0.299, 0.587, 0.114 ) );

    int lLumaAsAlpha = 0;

    if( uboParams.grayscaleRendering == 1.0f )
        outFragcolor = vec4( vec3(lLuma), (lLumaAsAlpha == 0 ) ? 1.0 : lLuma);
    else
        outFragcolor = vec4( lOutColor, (lLumaAsAlpha == 0 ) ? 1.0 : lLuma );

    // if( uboParams.grayscaleRendering == 1.0f ) outColor = vec4( vec3( dot( outColor.xyz, vec3( 0.2126, 0.7152, 0.0722 ) ) ), lBaseColor.a );
    // outColor = vec4(vec3((length(inWorldPos) - 0.5)), 1.0);
}
