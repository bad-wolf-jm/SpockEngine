

vec3 getNormalFromMap( sampler2D aNormalSampler, vec2 aCoords )
{
    // Perturb normal, see http://www.thetenthplanet.de/archives/1180
    vec3 tangentNormal = normalize( texture( aNormalSampler, aCoords ).xyz * 2.0 - vec3( 1.0 ) );

    vec3 dp1  = dFdx( inWorldPos );
    vec3 dp2  = dFdy( inWorldPos );
    vec2 duv1 = dFdx( aCoords );
    vec2 duv2 = dFdy( aCoords );

    // solve the linear system
    vec3 dp1perp = cross( inNormal, dp1 );
    vec3 dp2perp = cross( dp2, inNormal );
    vec3 T       = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B       = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame
    float invmax = inversesqrt( max( dot( T, T ), dot( B, B ) ) );

    return normalize( mat3( T * invmax, B * invmax, inNormal ) * tangentNormal );
}