
#version 450
#extension GL_GOOGLE_include_directive : require

layout( location = 0 ) in vec2 inUV;
layout( location = 1 ) in vec4 inConsoleUV;

layout( set = 0, binding = 0 ) uniform sampler2D sImage;

layout( location = 0 ) out vec4 outFragcolor;

void main()
{
    outFragcolor = texture(sImage, inUV);
}
