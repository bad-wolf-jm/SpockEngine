#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

LAYOUT_LOCATION( 0 ) in float3 inPos;
LAYOUT_LOCATION( 1 ) in float3 inNormal;

// Clever way of packing two UV channels into a single shader input
#if defined( MATERIAL_HAS_UV0 ) && !defined( MATERIAL_HAS_UV1 )
LAYOUT_LOCATION( 2 ) in float2 inUV0;
#else
LAYOUT_LOCATION( 2 ) in float4 inUV0;
#endif
