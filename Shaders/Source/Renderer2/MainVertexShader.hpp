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

void main()
{
    // Pass vertex-level data to the fragment shader
    outWorldPos = inPos;
    outNormal   = inNormal;
    outUV       = inUV;

    // Vertices are already transformed eiher using CUDA or a compute shader
    gl_Position = ubo.projection * ubo.view * vec4( inPos.xyz, 1.0 );
}