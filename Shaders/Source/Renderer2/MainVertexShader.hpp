#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "Varying.hpp"
#endif

LAYOUT_LOCATION( 0 ) __SHADER_OUTPUT__ float3 outWorldPos;

#if defined( MATERIAL_HAS_NORMALS )
LAYOUT_LOCATION( 1 ) __SHADER_OUTPUT__ float3 outNormal;
#endif

#if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
LAYOUT_LOCATION( 2 ) __SHADER_OUTPUT__ float2 outUV;
#elif defined( MATERIAL_HAS_UV1 )
LAYOUT_LOCATION( 2 ) __SHADER_OUTPUT__ float4 outUV;
#endif


// clang-format off
LAYOUT_UNIFORM( VIEW_PARAMETERS_BIND_POINT, 0 ) ViewParameters 
{ 
    float4x4 mProjection;
    float4x4 mView;
    float3   mCameraPosition;
} gView;
// clang-format on


void main()
{
    // Pass vertex-level data to the fragment shader
    outWorldPos = inPos;
    outNormal   = inNormal;
    outUV       = inUV;

    // Vertices are already transformed eiher using CUDA or a compute shader
    gl_Position = gView.mProjection * gView.mView * vec4( inPos.xyz, 1.0 );
}