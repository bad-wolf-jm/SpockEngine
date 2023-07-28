#ifndef _DEFINITIONS_H_
#define _DEFINITIONS_H_

#if defined( VULKAN_SEMANTICS )
#    define LAYOUT_LOCATION( x )          layout( location = x )
#    define LAYOUT_UNIFORM( s, b )        layout( set = s, binding = b ) uniform
#    define LAYOUT_UNIFORM_BUFFER( s, b ) layout( set = s, binding = b ) readonly buffer
#    define __UNIFORM__
#    define __UNIFORM_BUFFER__
#else
#    define LAYOUT_LOCATION( x )
#    define LAYOUT_UNIFORM( s, b ) struct
#    define LAYOUT_UNIFORM_BUFFER( s, b )
#    define __UNIFORM__
#    define __UNIFORM_BUFFER__ struct
#endif

#if defined( __GLSL__ ) || defined( __cplusplus )
#    define float2   vec2
#    define float3   vec3
#    define float4   vec4
#    define float3x3 mat3
#    define float4x4 mat4
#    if defined( __cplusplus )
typedef struct _sampler2D sampler2D;
#    endif
#endif

#if defined( __cplusplus )
#    include "Core/Math/Types.h"
inline float4 gl_Position;
#endif

#if defined( __cplusplus )
#    define ALIGN( x ) alignas( x )
#else
#    define ALIGN( x )
#endif

#define VIEW_PARAMETERS_BIND_POINT    0
#define CAMERA_PARAMETERS_BIND_POINT  1
#define MATERIAL_DATA_BIND_POINT      2
#define MATERIAL_TEXTURES_BIND_POINT  3
#define DIRECTIONAL_LIGHTS_BIND_POINT 4
#define PUNCTUAL_LIGHTS_BIND_POINT    5

#endif
