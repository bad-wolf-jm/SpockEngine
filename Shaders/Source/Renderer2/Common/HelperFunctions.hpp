#if defined( __cplusplus )
#    include "Common/Definitions.h"
#endif

float4 SRGBtoLINEAR( float4 srgbIn )
{
    float3 bLess = step( float3( 0.04045 ), srgbIn.xyz );
    float3 linOut =
        mix( srgbIn.xyz / float3( 12.92 ), pow( ( srgbIn.xyz + float3( 0.055 ) ) / float3( 1.055 ), float3( 2.4 ) ), bLess );

    return float4( linOut, srgbIn.w );
}

// float3 GetNormalFromMap( sampler2D aNormalSampler, float2 aCoords )
// {
//     // // Perturb normal, see http://www.thetenthplanet.de/archives/1180
//     float3 tangentNormal = texture( aNormalSampler, aCoords ).xyz * 2.0 - float3( 1.0 );

//     float3   q1  = dFdx( inWorldPos );
//     float3   q2  = dFdy( inWorldPos );
//     float2   st1 = dFdx( inUV.xy );
//     float2   st2 = dFdy( inUV.xy );
//     float3   N   = normalize( inNormal );
//     float3   T   = normalize( q1 * st2.t - q2 * st1.t );
//     float3   B   = -normalize( cross( N, T ) );
//     float3x3 TBN = float3x3( T, B, N );

//     return normalize( TBN * tangentNormal );
// }

float3 GetNormalFromMap( int aTexID, int aUVChannel )
{
    // // Perturb normal, see http://www.thetenthplanet.de/archives/1180
#if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
    float3 tangentNormal = texture( gTextures[aTexID], inUV ).xyz * 2.0 - float3( 1.0 );
#else
    float3 tangentNormal = texture( gTextures[aTexID], ( aUVChannel == 0 ) ? inUV.xy : inUV.zw ).xyz * 2.0 - float3( 1.0 );
#endif
    float3   q1  = dFdx( inWorldPos );
    float3   q2  = dFdy( inWorldPos );
    float2   st1 = dFdx( inUV.xy );
    float2   st2 = dFdy( inUV.xy );
    float3   N   = normalize( inNormal );
    float3   T   = normalize( q1 * st2.t - q2 * st1.t );
    float3   B   = -normalize( cross( N, T ) );
    float3x3 TBN = float3x3( T, B, N );

    return normalize( TBN * tangentNormal );
}
