#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    define __SHADER_INPUT__
#    define __SHADER_OUTPUT__
#else
#    define __SHADER_INPUT__  in
#    define __SHADER_OUTPUT__ out
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
