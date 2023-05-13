#version 450

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec4 outConsoleUV;

vec4 sGridPlane[6] = vec4[]( vec4( 1, 1, 0, 1 ), vec4( -1, -1, 0, 1 ), vec4( -1, 1, 0, 1 ), vec4( -1, -1, 0, 1 ), vec4( 1, 1, 0, 1 ), vec4( 1, -1, 0, 1 ) );


void main() 
{
    outUV = (sGridPlane[gl_VertexIndex].xy + vec2(1.0f, 1.0f)) / 2.0f;
    outConsoleUV = vec4(0.0, 0.0, 0.0, 0.0);

    gl_Position = sGridPlane[gl_VertexIndex];
}
