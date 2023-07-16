
#if defined( VULKAN_SEMANTICS )
#    define LAYOUT_LOCATION( x ) layout( location = x )
#    define LAYOUT_UNIFORM( s, b ) layout( set = s, binding = b )
#else
#    define LAYOUT_LOCATION( x )
#endif

#if defined(__cplusplus)
#include "Core/Math/Types.h"
float4 gl_Position;
#endif

#if defined( __GLSL__ ) || defined(__cplusplus)
#    define float2   vec2
#    define float3   vec3
#    define float4   vec4
#    define float3x3 mat3
#    define float4x4 mat4
#endif