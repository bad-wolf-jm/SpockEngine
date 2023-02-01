
#version 450
#extension GL_GOOGLE_include_directive : require

layout( location = 0 ) in vec2 inUV;
layout( location = 1 ) in vec4 inConsoleUV;

layout( set = 1, binding = 0 ) uniform sampler2D sImage;

void main()
{
    vec4 lOutColor = texture(sImage, inUV);
}
