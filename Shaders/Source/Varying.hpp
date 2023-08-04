#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

LAYOUT_LOCATION( 0 ) __SHADER_INPUT__ float3 inPos;

#if defined( MATERIAL_HAS_NORMALS )
LAYOUT_LOCATION( 1 ) __SHADER_INPUT__ float3 inNormal;
#endif

// Clever way of packing two UV channels into a single shader input
#if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
LAYOUT_LOCATION( 2 ) __SHADER_INPUT__ float2 inUV;
#elif defined( MATERIAL_HAS_UV1 )
LAYOUT_LOCATION( 2 ) __SHADER_INPUT__ float4 inUV;
#endif

LAYOUT_LOCATION( 3 ) __SHADER_INPUT__ float4 inUnused0;
LAYOUT_LOCATION( 4 ) __SHADER_INPUT__ float4 inUnused1;
