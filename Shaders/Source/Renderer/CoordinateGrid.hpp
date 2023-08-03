#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

#if defined( COORDINATE_GRID_VERTEX_SHADER )
// clang-format off
LAYOUT_UNIFORM( 0, 0 ) ViewUniforms
{
    float4x4 mViewMatrix;
    float4x4 mProjectionMatrix;
} gView;
// clang-format on
#endif

#if defined( COORDINATE_GRID_VERTEX_SHADER )
#    define INOUT __SHADER_OUTPUT__
#elif defined( COORDINATE_GRID_FRAGMENT_SHADER )
#    define INOUT __SHADER_INPUT__
#else
#    define INOUT
#endif

LAYOUT_LOCATION( 0 ) INOUT float gNear; // 0.01
LAYOUT_LOCATION( 1 ) INOUT float gFar;  // 100
LAYOUT_LOCATION( 2 ) INOUT float3 gNearPoint;
LAYOUT_LOCATION( 3 ) INOUT float3 gFarPoint;
LAYOUT_LOCATION( 4 ) INOUT float4x4 gFragView;
LAYOUT_LOCATION( 8 ) INOUT float4x4 gFragProj;

#if defined( COORDINATE_GRID_FRAGMENT_SHADER )
LAYOUT_LOCATION( 0 ) __SHADER_OUTPUT__ float4 outColor;
#endif

// Grid position are in clipped space
#if defined( COORDINATE_GRID_VERTEX_SHADER )
float4 gridPlane[6] = float4[]( float4( 1, 1, 0, 1 ), float4( -1, -1, 0, 1 ), float4( -1, 1, 0, 1 ), float4( -1, -1, 0, 1 ),
                                float4( 1, 1, 0, 1 ), float4( 1, -1, 0, 1 ) );

float3 UnprojectPoint( float x, float y, float z, float4x4 aView, float4x4 aProjection )
{
    float4x4 viewInv          = inverse( aView );
    float4x4 projInv          = inverse( aProjection );
    float4   unprojectedPoint = viewInv * projInv * float4( x, y, z, 1.0 );

    return unprojectedPoint.xyz / unprojectedPoint.w;
}
#endif

#if defined( COORDINATE_GRID_FRAGMENT_SHADER )
float4 grid( float3 fragPos3D, float scale, bool drawAxis )
{
    vec2   coord      = fragPos3D.xz * scale;
    vec2   derivative = fwidth( coord );
    vec2   grid       = abs( fract( coord - 0.5 ) - 0.5 ) / derivative;
    float  line       = min( grid.x, grid.y );
    float  minimumz   = min( derivative.y, 1 );
    float  minimumx   = min( derivative.x, 1 );
    float4 color      = float4( 1.0, 1.0, 1.0, 1.0 - min( line, 1.0 ) );

    // z axis
    if( fragPos3D.x > -minimumx && fragPos3D.x < minimumx )
        color = float4( 0.0, 0.0, 1.0, 1.0 - min( line, 1.0 ) );

    // x axis
    if( fragPos3D.z > -minimumz && fragPos3D.z < minimumz )
        color = float4( 1.0, 0.0, 0.0, 1.0 - min( line, 1.0 ) );

    return color;
}

float computeDepth( float3 pos )
{
    float4 clip_space_pos = gFragProj * gFragView * float4( pos.xyz, 1.0 );
    return ( clip_space_pos.z / clip_space_pos.w );
}

float computeLinearDepth( float3 pos )
{
    float4 clip_space_pos   = gFragProj * gFragView * float4( pos.xyz, 1.0 );
    float  clip_space_depth = ( clip_space_pos.z / clip_space_pos.w ) * 2.0 - 1.0;       // put back between -1 and 1
    float  linearDepth =
        ( 2.0 * gNear * gFar ) / ( gFar + gNear - clip_space_depth * ( gFar - gNear ) ); // get linear value between 0.01 and 100

    return linearDepth / gFar;
}
#endif

void main()
{
#if defined( COORDINATE_GRID_VERTEX_SHADER )
    float3 p    = gridPlane[gl_VertexIndex].xyz;
    gNear       = 0.01;
    gFar        = 25.0f;
    gFragView   = gView.mViewMatrix;
    gFragProj   = gView.mProjectionMatrix;
    gNearPoint  = UnprojectPoint( p.x, p.y, 0.0, gView.mViewMatrix, gView.mProjectionMatrix ).xyz; // unprojecting on the inNear plane
    gFarPoint   = UnprojectPoint( p.x, p.y, 1.0, gView.mViewMatrix, gView.mProjectionMatrix ).xyz; // unprojecting on the inFar plane
    gl_Position = float4( p, 1.0 ); // using directly the clipped coordinates
#endif

#if defined( COORDINATE_GRID_FRAGMENT_SHADER )
    float  t          = -gNearPoint.y / ( gFarPoint.y - gNearPoint.y );
    float3 fragPos3D  = gNearPoint + t * ( gFarPoint - gNearPoint );
    gl_FragDepth      = computeDepth( fragPos3D );
    float linearDepth = computeLinearDepth( fragPos3D );
    float fading      = max( 0, ( .50 - linearDepth ) );
    outColor          = ( grid( fragPos3D, 1, true ) ) * float( t > 0 ); // adding multiple resolution for the grid
    outColor.a *= fading;
#endif
}