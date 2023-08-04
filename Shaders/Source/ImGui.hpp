#if defined( __cplusplus )
#    include "Common/Definitions.hpp"
#endif

#if defined( IMGUI_VERTEX_SHADER )
#    define INOUT __SHADER_OUTPUT__
#elif defined( IMGUI_FRAGMENT_SHADER )
#    define INOUT __SHADER_INPUT__
#else
#    define INOUT
#endif

#if defined( IMGUI_VERTEX_SHADER )
LAYOUT_LOCATION( 0 ) __SHADER_INPUT__ float2 aPos;
LAYOUT_LOCATION( 1 ) __SHADER_INPUT__ float2 aUV;
LAYOUT_LOCATION( 2 ) __SHADER_INPUT__ float4 aColor;

layout( push_constant ) uniform uPushConstant
{
    float2 uScale;
    float2 uTranslate;
}
pc;

__SHADER_OUTPUT__ gl_PerVertex
{
    float4 gl_Position;
};
#endif

LAYOUT_LOCATION( 0 ) INOUT struct
{
    float4 Color;
    float2 UV;
} Out;

#if defined( IMGUI_FRAGMENT_SHADER )
LAYOUT_UNIFORM( 0, 0 ) sampler2D sTexture;
LAYOUT_LOCATION( 0 ) __SHADER_OUTPUT__ float4 fColor;
#endif

void main()
{
#if defined( IMGUI_VERTEX_SHADER )
    Out.Color   = aColor;
    Out.UV      = aUV;
    gl_Position = float4( aPos * pc.uScale + pc.uTranslate, 0, 1 );
#endif

#if defined( IMGUI_FRAGMENT_SHADER )
    fColor = Out.Color * texture( sTexture, Out.UV.st );
#endif
}
