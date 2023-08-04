#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#    include "Varying.hpp"
#endif

#if defined( PUNCTUAL_LIGHT_SHADOW_VERTEX_SHADER )
LAYOUT_LOCATION( 0 ) __SHADER_OUTPUT__ float4 outPos;
LAYOUT_LOCATION( 1 ) __SHADER_OUTPUT__ float3 outLightPos;

LAYOUT_UNIFORM( 0, 0 ) UBO
{
    float4x4 depthMVP;
    float4   lightPos;
}
ubo;
#endif

#if defined( DIRECTIONAL_LIGHT_SHADOW_VERTEX_SHADER )
LAYOUT_UNIFORM( 0, 0 ) UBO
{
    float4x4 depthMVP;
}
ubo;
#endif

#if defined( PUNCTUAL_LIGHT_SHADOW_FRAGMENT_SHADER )
LAYOUT_LOCATION( 0 ) __SHADER_INPUT__ float4 inPos;
LAYOUT_LOCATION( 1 ) __SHADER_INPUT__ float3 inLightPos;
LAYOUT_LOCATION( 0 ) __SHADER_OUTPUT__ float4 outFragColor;
#endif

void main()
{
#if defined( DIRECTIONAL_LIGHT_SHADOW_VERTEX_SHADER )
    gl_Position = ubo.depthMVP * float4( inPos, 1.0 );
#endif

#if defined( PUNCTUAL_LIGHT_SHADOW_VERTEX_SHADER )
    gl_Position = ubo.depthMVP * float4( inPos, 1.0 );
    outPos      = float4( inPos, 1.0 );
    outLightPos = ubo.lightPos.xyz;
#endif

#if defined( PUNCTUAL_LIGHT_SHADOW_FRAGMENT_SHADER )
    vec3 lightVec = inPos.xyz - inLightPos;
    outFragColor  = float4( dot( lightVec, lightVec ), 0.0, 0.0, 1.0 );
#endif
}