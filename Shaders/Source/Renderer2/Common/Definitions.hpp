
#if defined( VULKAN_SEMANTICS )
#    define LAYOUT_LOCATION( x )          layout( location = x )
#    define LAYOUT_UNIFORM( s, b )        layout( set = s, binding = b )
#    define LAYOUT_UNIFORM_BUFFER( s, b ) layout( set = s, binding = b ) readonly buffer
#    define __UNIFORM__                   uniform
#    define __UNIFORM_BUFFER__            readonly buffer
#else
#    define LAYOUT_LOCATION( x )
#    define LAYOUT_UNIFORM( s, b )
#    define LAYOUT_UNIFORM_BUFFER( s, b )
#    define __UNIFORM__
#    define __UNIFORM_BUFFER__            struct
#endif

#if defined( __GLSL__ ) || defined( __cplusplus )
#    define float2   vec2
#    define float3   vec3
#    define float4   vec4
#    define float3x3 mat3
#    define float4x4 mat4
#    if defined( __cplusplus )
typedef struct sampler2D;
#    endif
#endif

#if defined( __cplusplus )
#    include "Core/Math/Types.h"
float4 gl_Position;
#endif
