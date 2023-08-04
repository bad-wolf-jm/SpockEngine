//------------------------------------------------------------------------------
// Shadow Sampling Types
//------------------------------------------------------------------------------

// Keep this in sync with PerViewUniforms.h
#define SHADOW_SAMPLING_RUNTIME_PCF  0u
#define SHADOW_SAMPLING_RUNTIME_EVSM 1u
#define SHADOW_SAMPLING_RUNTIME_DPCF 2u
#define SHADOW_SAMPLING_RUNTIME_PCSS 3u

// TODO: this should be user-settable, maybe at the material level
#define SHADOW_SAMPLING_PCF_HARD 0
#define SHADOW_SAMPLING_PCF_LOW  1
#define SHADOW_SAMPLING_METHOD   SHADOW_SAMPLING_PCF_LOW

//------------------------------------------------------------------------------
// PCF Shadow Sampling
//------------------------------------------------------------------------------

float sampleDepth( const sampler2DArrayShadow map, const float4 scissorNormalized, const uint layer, float2 uv, float depth )
{

    // clamp needed for directional lights and/or large kernels
    uv = clamp( uv, scissorNormalized.xy, scissorNormalized.zw );

    // depth must be clamped to support floating-point depth formats which are always in
    // the range [0, 1].
    return texture( map, float4( uv, layer, saturate( depth ) ) );
}

#if SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_HARD
// use hardware assisted PCF
float ShadowSample_PCF_Hard( const sampler2DArrayShadow map, const float4 scissorNormalized, const uint layer,
                             const float4 shadowPosition )
{
    float3 position = shadowPosition.xyz * ( 1.0 / shadowPosition.w );
    // note: shadowPosition.z is in the [1, 0] range (reversed Z)
    return sampleDepth( map, scissorNormalized, layer, position.xy, position.z );
}
#endif

#if SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_LOW
// use hardware assisted PCF + 3x3 gaussian filter
float ShadowSample_PCF_Low( const sampler2DArrayShadow map, const float4 scissorNormalized, const uint layer,
                            const float4 shadowPosition )
{
    float3 position = shadowPosition.xyz * ( 1.0 / shadowPosition.w );
    // note: shadowPosition.z is in the [1, 0] range (reversed Z)
    float2 size      = float2( textureSize( map, 0 ) );
    float2 texelSize = float2( 1.0 ) / size;

    //  Castaño, 2013, "Shadow Mapping Summary Part 1"
    float depth = position.z;

    // clamp position to avoid overflows below, which cause some GPUs to abort
    position.xy = clamp( position.xy, float2( -1.0 ), float2( 2.0 ) );

    float2 offset = float2( 0.5 );
    float2 uv     = ( position.xy * size ) + offset;
    float2 base   = ( floor( uv ) - offset ) * texelSize;
    float2 st     = fract( uv );

    float2 uw = float2( 3.0 - 2.0 * st.x, 1.0 + 2.0 * st.x );
    float2 vw = float2( 3.0 - 2.0 * st.y, 1.0 + 2.0 * st.y );

    float2 u = float2( ( 2.0 - st.x ) / uw.x - 1.0, st.x / uw.y + 1.0 );
    float2 v = float2( ( 2.0 - st.y ) / vw.x - 1.0, st.y / vw.y + 1.0 );

    u *= texelSize.x;
    v *= texelSize.y;

    float sum = 0.0;
    sum += uw.x * vw.x * sampleDepth( map, scissorNormalized, layer, base + float2( u.x, v.x ), depth );
    sum += uw.y * vw.x * sampleDepth( map, scissorNormalized, layer, base + float2( u.y, v.x ), depth );
    sum += uw.x * vw.y * sampleDepth( map, scissorNormalized, layer, base + float2( u.x, v.y ), depth );
    sum += uw.y * vw.y * sampleDepth( map, scissorNormalized, layer, base + float2( u.y, v.y ), depth );
    return sum * ( 1.0 / 16.0 );
}
#endif

// use manual PCF
float ShadowSample_PCF( const sampler2DArray map, const float4 scissorNormalized, const uint layer, const float4 shadowPosition )
{
    float3 position = shadowPosition.xyz * ( 1.0 / shadowPosition.w );
    // note: shadowPosition.z is in the [1, 0] range (reversed Z)
    float2 size = float2( textureSize( map, 0 ) );
    float2 tc   = clamp( position.xy, scissorNormalized.xy, scissorNormalized.zw );
    float2 st   = tc.xy * size - 0.5;

    float4 d;
#if defined( FILAMENT_HAS_FEATURE_TEXTURE_GATHER )
    d = textureGather( map, float3( tc, layer ), 0 ); // 01, 11, 10, 00
#else
    // we must use texelFetchOffset before texelLodOffset filters
    d[0] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 0, 1 ) ).r;
    d[1] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 1, 1 ) ).r;
    d[2] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 1, 0 ) ).r;
    d[3] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 0, 0 ) ).r;
#endif

    float4 pcf  = step( 0.0, position.zzzz - d );
    float2 grad = fract( st );
    return mix( mix( pcf.w, pcf.z, grad.x ), mix( pcf.x, pcf.y, grad.x ), grad.y );
}

//------------------------------------------------------------------------------
// DPCF sampling
//------------------------------------------------------------------------------

// Poisson disk generated with 'poisson-disk-generator' tool from
// https://github.com/corporateshark/poisson-disk-generator by Sergey Kosarevsky
/*const*/ float2 poissonDisk[64] = float2[]( // don't use 'const' b/c of OSX GL compiler bug
    float2( 0.511749, 0.547686 ), float2( 0.58929, 0.257224 ), float2( 0.165018, 0.57663 ), float2( 0.407692, 0.742285 ),
    float2( 0.707012, 0.646523 ), float2( 0.31463, 0.466825 ), float2( 0.801257, 0.485186 ), float2( 0.418136, 0.146517 ),
    float2( 0.579889, 0.0368284 ), float2( 0.79801, 0.140114 ), float2( -0.0413185, 0.371455 ), float2( -0.0529108, 0.627352 ),
    float2( 0.0821375, 0.882071 ), float2( 0.17308, 0.301207 ), float2( -0.120452, 0.867216 ), float2( 0.371096, 0.916454 ),
    float2( -0.178381, 0.146101 ), float2( -0.276489, 0.550525 ), float2( 0.12542, 0.126643 ), float2( -0.296654, 0.286879 ),
    float2( 0.261744, -0.00604975 ), float2( -0.213417, 0.715776 ), float2( 0.425684, -0.153211 ), float2( -0.480054, 0.321357 ),
    float2( -0.0717878, -0.0250567 ), float2( -0.328775, -0.169666 ), float2( -0.394923, 0.130802 ), float2( -0.553681, -0.176777 ),
    float2( -0.722615, 0.120616 ), float2( -0.693065, 0.309017 ), float2( 0.603193, 0.791471 ), float2( -0.0754941, -0.297988 ),
    float2( 0.109303, -0.156472 ), float2( 0.260605, -0.280111 ), float2( 0.129731, -0.487954 ), float2( -0.537315, 0.520494 ),
    float2( -0.42758, 0.800607 ), float2( 0.77309, -0.0728102 ), float2( 0.908777, 0.328356 ), float2( 0.985341, 0.0759158 ),
    float2( 0.947536, -0.11837 ), float2( -0.103315, -0.610747 ), float2( 0.337171, -0.584 ), float2( 0.210919, -0.720055 ),
    float2( 0.41894, -0.36769 ), float2( -0.254228, -0.49368 ), float2( -0.428562, -0.404037 ), float2( -0.831732, -0.189615 ),
    float2( -0.922642, 0.0888026 ), float2( -0.865914, 0.427795 ), float2( 0.706117, -0.311662 ), float2( 0.545465, -0.520942 ),
    float2( -0.695738, 0.664492 ), float2( 0.389421, -0.899007 ), float2( 0.48842, -0.708054 ), float2( 0.760298, -0.62735 ),
    float2( -0.390788, -0.707388 ), float2( -0.591046, -0.686721 ), float2( -0.769903, -0.413775 ), float2( -0.604457, -0.502571 ),
    float2( -0.557234, 0.00451362 ), float2( 0.147572, -0.924353 ), float2( -0.0662488, -0.892081 ), float2( 0.863832, -0.407206 ) );

// tap count up can go up to 64
const uint DPCF_SHADOW_TAP_COUNT = 12u;
// more samples lead to better "shape" of the hardened shadow
const uint PCSS_SHADOW_BLOCKER_SEARCH_TAP_COUNT = 16u;
// less samples lead to noisier shadows (can be mitigated with TAA)
const uint PCSS_SHADOW_FILTER_TAP_COUNT = 16u;

float hardenedKernel( float x )
{
    // this is basically a stronger smoothstep()
    x       = 2.0 * x - 1.0;
    float s = sign( x );
    x       = 1.0 - s * x;
    x       = x * x * x;
    x       = s - x * s;

    return 0.5 * x + 0.5;
}

float2 computeReceiverPlaneDepthBias( const float3 position )
{
    // see: GDC '06: Shadow Mapping: GPU-based Tips and Techniques
    // Chain rule to compute dz/du and dz/dv
    // |dz/du|   |du/dx du/dy|^-T   |dz/dx|
    // |dz/dv| = |dv/dx dv/dy|    * |dz/dy|
    float3 duvz_dx = dFdx( position );
    float3 duvz_dy = dFdy( position );
    float2 dz_duv  = inverse( transpose( float2x2( duvz_dx.xy, duvz_dy.xy ) ) ) * float2( duvz_dx.z, duvz_dy.z );
    return dz_duv;
}

float2x2 getRandomRotationMatrix( float2 fragCoord )
{
    // rotate the poisson disk randomly
    fragCoord += float2( frameUniforms.temporalNoise ); // 0 when TAA is not used
    float    randomAngle = interleavedGradientNoise( fragCoord ) * ( 2.0 * PI );
    float2   randomBase  = float2( cos( randomAngle ), sin( randomAngle ) );
    float2x2 R           = float2x2( randomBase.x, randomBase.y, -randomBase.y, randomBase.x );
    return R;
}

float getPenumbraLs( const bool DIRECTIONAL, const int index, const float zLight )
{
    float penumbra;
    // This conditional is resolved at compile time
    if( DIRECTIONAL )
    {
        penumbra = shadowUniforms.shadows[index].bulbRadiusLs;
    }
    else
    {
        // the penumbra radius depends on the light-space z for spotlights
        penumbra = shadowUniforms.shadows[index].bulbRadiusLs / zLight;
    }
    return penumbra;
}

float getPenumbraRatio( const bool DIRECTIONAL, const int index, float z_receiver, float z_blocker )
{
    // z_receiver/z_blocker are not linear depths (i.e. they're not distances)
    // Penumbra ratio for PCSS is given by:  pr = (d_receiver - d_blocker) / d_blocker
    float penumbraRatio;
    if( DIRECTIONAL )
    {
        // TODO: take lispsm into account
        // For directional lights, the depths are linear but depend on the position (because of LiSPSM).
        // With:        z_linear = f + z * (n - f)
        // We get:      (r-b)/b ==> (f/(n-f) + r_linear) / (f/(n-f) + b_linear) - 1
        // Assuming f>>n and ignoring LISPSM, we get:
        penumbraRatio = ( z_blocker - z_receiver ) / ( 1.0 - z_blocker );
    }
    else
    {
        // For spotlights, the depths are congruent to 1/z, specifically:
        //      z_linear = (n * f) / (n + z * (f - n))
        // replacing in (r - b) / b gives:
        float nearOverFarMinusNear = shadowUniforms.shadows[index].nearOverFarMinusNear;
        penumbraRatio              = ( nearOverFarMinusNear + z_blocker ) / ( nearOverFarMinusNear + z_receiver ) - 1.0;
    }
    return penumbraRatio * frameUniforms.shadowPenumbraRatioScale;
}

void blockerSearchAndFilter( out float occludedCount, out float z_occSum, const sampler2DArray map, const float4 scissorNormalized,
                             const float2 uv, const float z_rec, const uint layer, const float2 filterRadii, const float2x2 R,
                             const float2 dz_duv, const uint tapCount )
{
    occludedCount = 0.0;
    z_occSum      = 0.0;
    for( uint i = 0u; i < tapCount; i++ )
    {
        float2 duv = R * ( poissonDisk[i] * filterRadii );
        float2 tc  = clamp( uv + duv, scissorNormalized.xy, scissorNormalized.zw );

        float z_occ = textureLod( map, float3( tc, layer ), 0.0 ).r;

        // note: z_occ and z_rec are not necessarily linear here, comparing them is always okay for
        // the regular PCF, but the "distance" is meaningless unless they are actually linear
        // (e.g.: for the directional light).
        // Either way, if we assume that all the samples are close to each other we can take their
        // average regardless, and the average depth value of the occluders
        // becomes: z_occSum / occludedCount.

        // receiver plane depth bias
        float z_bias   = dot( dz_duv, duv );
        float dz       = z_occ - z_rec; // dz>0 when blocker is between receiver and light
        float occluded = step( z_bias, dz );
        occludedCount += occluded;
        z_occSum += z_occ * occluded;
    }
}

float filterPCSS( const sampler2DArray map, const float4 scissorNormalized, const float2 size, const float2 uv, const float z_rec,
                  const uint layer, const float2 filterRadii, const float2x2 R, const float2 dz_duv, const uint tapCount )
{

    float occludedCount = 0.0;
    for( uint i = 0u; i < tapCount; i++ )
    {
        float2 duv = R * ( poissonDisk[i] * filterRadii );

        // sample the shadow map with a 2x2 PCF, this helps a lot in low resolution areas
        float4 d;
        float2 tc   = clamp( uv + duv, scissorNormalized.xy, scissorNormalized.zw );
        float2 st   = tc.xy * size - 0.5;
        float2 grad = fract( st );

#if defined( FILAMENT_HAS_FEATURE_TEXTURE_GATHER )
        d = textureGather( map, float3( tc, layer ), 0 ); // 01, 11, 10, 00
#else
        // we must use texelFetchOffset before texelLodOffset filters
        d[0] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 0, 1 ) ).r;
        d[1] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 1, 1 ) ).r;
        d[2] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 1, 0 ) ).r;
        d[3] = texelFetchOffset( map, ivec3( st, layer ), 0, ivec2( 0, 0 ) ).r;
#endif

        // receiver plane depth bias
        float  z_bias = dot( dz_duv, duv );
        float4 dz     = d - float4( z_rec ); // dz>0 when blocker is between receiver and light
        float4 pcf    = step( z_bias, dz );
        occludedCount += mix( mix( pcf.w, pcf.z, grad.x ), mix( pcf.x, pcf.y, grad.x ), grad.y );
    }
    return occludedCount * ( 1.0 / float( tapCount ) );
}

/*
 * DPCF, PCF with contact hardenning simulation.
 * see "Shadow of Cold War", A scalable approach to shadowing -- by Kevin Myers
 */
float ShadowSample_DPCF( const bool DIRECTIONAL, const sampler2DArray map, const float4 scissorNormalized, const uint layer,
                         const int index, const float4 shadowPosition, const float zLight )
{
    float3 position  = shadowPosition.xyz * ( 1.0 / shadowPosition.w );
    float2 texelSize = float2( 1.0 ) / float2( textureSize( map, 0 ) );

    // We need to use the shadow receiver plane depth bias to combat shadow acne due to the
    // large kernel.
    float2 dz_duv = computeReceiverPlaneDepthBias( position );

    float penumbra = getPenumbraLs( DIRECTIONAL, index, zLight );

    // rotate the poisson disk randomly
    float2x2 R = getRandomRotationMatrix( gl_FragCoord.xy );

    float occludedCount = 0.0;
    float z_occSum      = 0.0;

    blockerSearchAndFilter( occludedCount, z_occSum, map, scissorNormalized, position.xy, position.z, layer, texelSize * penumbra, R,
                            dz_duv, DPCF_SHADOW_TAP_COUNT );

    // early exit if there is no occluders at all, also avoids a divide-by-zero below.
    if( z_occSum == 0.0 )
    {
        return 1.0;
    }

    float penumbraRatio = getPenumbraRatio( DIRECTIONAL, index, position.z, z_occSum / occludedCount );

    // The main way we're diverging from PCSS is that we're not going to sample again, instead
    // we're going to reuse the blocker search samples and we're going to use the penumbra ratio
    // as a parameter to lerp between a hardened PCF kernel and the search PCF kernel.
    // We need a parameter to blend between the the "hardened" kernel and the "soft" kernel,
    // to this end clamp the penumbra ratio between 0 (blocker is close to the receiver) and
    // 1 (blocker is close to the light).
    penumbraRatio = saturate( penumbraRatio );

    // regular PCF weight (i.e. average of samples in shadow)
    float percentageOccluded = occludedCount * ( 1.0 / float( DPCF_SHADOW_TAP_COUNT ) );

    // now we just need to lerp between hardened PCF and regular PCF based on alpha
    percentageOccluded = mix( hardenedKernel( percentageOccluded ), percentageOccluded, penumbraRatio );
    return 1.0 - percentageOccluded;
}

float ShadowSample_PCSS( const bool DIRECTIONAL, const sampler2DArray map, const float4 scissorNormalized, const uint layer,
                         const int index, const float4 shadowPosition, const float zLight )
{
    float2 size      = float2( textureSize( map, 0 ) );
    float2 texelSize = float2( 1.0 ) / size;
    float3 position  = shadowPosition.xyz * ( 1.0 / shadowPosition.w );

    // We need to use the shadow receiver plane depth bias to combat shadow acne due to the
    // large kernel.
    float2 dz_duv = computeReceiverPlaneDepthBias( position );

    float penumbra = getPenumbraLs( DIRECTIONAL, index, zLight );

    // rotate the poisson disk randomly
    float2x2 R = getRandomRotationMatrix( gl_FragCoord.xy );

    float occludedCount = 0.0;
    float z_occSum      = 0.0;

    blockerSearchAndFilter( occludedCount, z_occSum, map, scissorNormalized, position.xy, position.z, layer, texelSize * penumbra, R,
                            dz_duv, PCSS_SHADOW_BLOCKER_SEARCH_TAP_COUNT );

    // early exit if there is no occluders at all, also avoids a divide-by-zero below.
    if( z_occSum == 0.0 )
    {
        return 1.0;
    }

    float penumbraRatio = getPenumbraRatio( DIRECTIONAL, index, position.z, z_occSum / occludedCount );

    float percentageOccluded = filterPCSS( map, scissorNormalized, size, position.xy, position.z, layer,
                                           texelSize * ( penumbra * penumbraRatio ), R, dz_duv, PCSS_SHADOW_FILTER_TAP_COUNT );

    return 1.0 - percentageOccluded;
}

//------------------------------------------------------------------------------
// Screen-space Contact Shadows
//------------------------------------------------------------------------------

struct ScreenSpaceRay
{
    float3 ssRayStart;
    float3 ssRayEnd;
    float3 ssViewRayEnd;
    float3 uvRayStart;
    float3 uvRay;
};

void initScreenSpaceRay( out ScreenSpaceRay ray, float3 wsRayStart, float3 wsRayDirection, float wsRayLength )
{
    float4x4 worldToClip = getClipFromWorldMatrix();
    float4x4 viewToClip  = getClipFromViewMatrix();

    // ray end in world space
    float3 wsRayEnd = wsRayStart + wsRayDirection * wsRayLength;

    // ray start/end in clip space (z is inverted: [1,0])
    float4 csRayStart   = worldToClip * float4( wsRayStart, 1.0 );
    float4 csRayEnd     = worldToClip * float4( wsRayEnd, 1.0 );
    float4 csViewRayEnd = csRayStart + viewToClip * float4( 0.0, 0.0, wsRayLength, 0.0 );

    // ray start/end in screen space (z is inverted: [1,0])
    ray.ssRayStart   = csRayStart.xyz * ( 1.0 / csRayStart.w );
    ray.ssRayEnd     = csRayEnd.xyz * ( 1.0 / csRayEnd.w );
    ray.ssViewRayEnd = csViewRayEnd.xyz * ( 1.0 / csViewRayEnd.w );

    // convert all to uv (texture) space (z is inverted: [1,0])
    float3 uvRayEnd = float3( ray.ssRayEnd.xy * 0.5 + 0.5, ray.ssRayEnd.z );
    ray.uvRayStart  = float3( ray.ssRayStart.xy * 0.5 + 0.5, ray.ssRayStart.z );
    ray.uvRay       = uvRayEnd - ray.uvRayStart;
}

float screenSpaceContactShadow( float3 lightDirection )
{
    // cast a ray in the direction of the light
    float occlusion    = 0.0;
    int   kStepCount   = ( frameUniforms.directionalShadows >> 8 ) & 0xFF;
    float kDistanceMax = frameUniforms.ssContactShadowDistance;

    ScreenSpaceRay rayData;
    initScreenSpaceRay( rayData, shading_position, lightDirection, kDistanceMax );

    // step
    float dt = 1.0 / float( kStepCount );

    // tolerance
    float tolerance = abs( rayData.ssViewRayEnd.z - rayData.ssRayStart.z ) * dt;

    // dither the ray with interleaved gradient noise
    float dither = interleavedGradientNoise( gl_FragCoord.xy ) - 0.5;

    // normalized position on the ray (0 to 1)
    float t = dt * dither + dt;

    float3 ray;
    for( int i = 0; i < kStepCount; i++, t += dt )
    {
        ray      = rayData.uvRayStart + rayData.uvRay * t;
        float z  = textureLod( light_structure, uvToRenderTargetUV( ray.xy ), 0.0 ).r;
        float dz = z - ray.z;
        if( abs( tolerance - dz ) < tolerance )
        {
            occlusion = 1.0;
            break;
        }
    }

    // we fade out the contribution of contact shadows towards the edge of the screen
    // because we don't have depth data there
    float2 fade = max( 12.0 * abs( ray.xy - 0.5 ) - 5.0, 0.0 );
    occlusion *= saturate( 1.0 - dot( fade, fade ) );
    return occlusion;
}

//------------------------------------------------------------------------------
// VSM
//------------------------------------------------------------------------------

float linstep( const float min, const float max, const float v )
{
    // we could use smoothstep() too
    return clamp( ( v - min ) / ( max - min ), 0.0, 1.0 );
}

float reduceLightBleed( const float pMax, const float amount )
{
    // Remove the [0, amount] tail and linearly rescale (amount, 1].
    return linstep( amount, 1.0, pMax );
}

float chebyshevUpperBound( const float2 moments, const float mean, const float minVariance, const float lightBleedReduction )
{
    // Donnelly and Lauritzen 2006, "Variance Shadow Maps"

    float variance = moments.y - ( moments.x * moments.x );
    variance       = max( variance, minVariance );

    float d    = mean - moments.x;
    float pMax = variance / ( variance + d * d );

    pMax = reduceLightBleed( pMax, lightBleedReduction );

    return mean <= moments.x ? 1.0 : pMax;
}

float evaluateShadowVSM( const float2 moments, const float depth )
{
    float depthScale  = frameUniforms.vsmDepthScale * depth;
    float minVariance = depthScale * depthScale;
    return chebyshevUpperBound( moments, depth, minVariance, frameUniforms.vsmLightBleedReduction );
}

float ShadowSample_VSM( const bool ELVSM, const sampler2DArray shadowMap, const float4 scissorNormalized, const uint layer,
                        const float4 shadowPosition )
{

    // note: shadowPosition.z is in linear light-space normalized to [0, 1]
    //  see: ShadowMap::computeVsmLightSpaceMatrix() in ShadowMap.cpp
    //  see: computeLightSpacePosition() in common_shadowing.fs
    float3 position = float3( shadowPosition.xy * ( 1.0 / shadowPosition.w ), shadowPosition.z );

    // Note: we don't need to clamp to `scissorNormalized` in the VSM case because this is only
    // needed when the shadow casters and receivers are different, which is never the case with VSM
    // (see ShadowMap.cpp).

    // Read the shadow map with all available filtering
    float4 moments = texture( shadowMap, float3( position.xy, layer ) );
    float  depth   = position.z;

    // EVSM depth warping
    depth = depth * 2.0 - 1.0;
    depth = frameUniforms.vsmExponent * depth;

    depth   = exp( depth );
    float p = evaluateShadowVSM( moments.xy, depth );
    if( ELVSM )
    {
        p = min( p, evaluateShadowVSM( moments.zw, -1.0 / depth ) );
    }
    return p;
}

//------------------------------------------------------------------------------
// Shadow sampling dispatch
//------------------------------------------------------------------------------

/**
 * Samples the light visibility at the specified position in light (shadow)
 * space. The output is a filtered visibility factor that can be used to multiply
 * the light intensity.
 */

// get texture coordinate for directional and spot shadow maps
#if defined( VARIANT_HAS_DIRECTIONAL_LIGHTING )
float4 getShadowPosition( const int cascade )
{
    return getCascadeLightSpacePosition( cascade );
}
#endif

#if defined( VARIANT_HAS_DYNAMIC_LIGHTING )
float4 getShadowPosition( const int index, const float3 dir, const float zLight )
{
    return getSpotLightSpacePosition( index, dir, zLight );
}
#endif

int getPointLightFace( const float3 r )
{
    float4 tc;
    float  rx = abs( r.x );
    float  ry = abs( r.y );
    float  rz = abs( r.z );
    float  d  = max( rx, max( ry, rz ) );
    if( d == rx )
    {
        return ( r.x >= 0.0 ? 0 : 1 );
    }
    else if( d == ry )
    {
        return ( r.y >= 0.0 ? 2 : 3 );
    }
    else
    {
        return ( r.z >= 0.0 ? 4 : 5 );
    }
}

// PCF sampling
float shadow( const bool DIRECTIONAL, const sampler2DArrayShadow shadowMap, const int index, float4 shadowPosition, float zLight )
{
    float4 scissorNormalized = shadowUniforms.shadows[index].scissorNormalized;
    uint   layer             = shadowUniforms.shadows[index].layer;
#if SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_HARD
    return ShadowSample_PCF_Hard( shadowMap, scissorNormalized, layer, shadowPosition );
#elif SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_LOW
    return ShadowSample_PCF_Low( shadowMap, scissorNormalized, layer, shadowPosition );
#endif
}

// Shadow requiring a sampler2D sampler (VSM, DPCF and PCSS)
float shadow( const bool DIRECTIONAL, const sampler2DArray shadowMap, const int index, float4 shadowPosition, float zLight )
{
    float4 scissorNormalized = shadowUniforms.shadows[index].scissorNormalized;
    uint   layer             = shadowUniforms.shadows[index].layer;
    // This conditional is resolved at compile time
    if( frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_EVSM )
    {
        bool elvsm = shadowUniforms.shadows[index].elvsm;
        return ShadowSample_VSM( elvsm, shadowMap, scissorNormalized, layer, shadowPosition );
    }

    if( frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_DPCF )
    {
        return ShadowSample_DPCF( DIRECTIONAL, shadowMap, scissorNormalized, layer, index, shadowPosition, zLight );
    }

    if( frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_PCSS )
    {
        return ShadowSample_PCSS( DIRECTIONAL, shadowMap, scissorNormalized, layer, index, shadowPosition, zLight );
    }

    if( frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_PCF )
    {
        // This is here mostly for debugging at this point.
        // Note: In this codepath, the normal bias is not applied because we're in the VSM variant.
        // (see: get{Cascade|Spot}LightSpacePosition)
        return ShadowSample_PCF( shadowMap, scissorNormalized, layer, shadowPosition );
    }

    // should not happen
    return 0.0;
}
