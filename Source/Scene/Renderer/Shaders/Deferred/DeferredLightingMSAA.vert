#version 450

layout (location = 0) out vec2 outUV;

vec4 gridPlane[6] = vec4[]( vec4( 1, 1, 0, 1 ), vec4( -1, -1, 0, 1 ), vec4( -1, 1, 0, 1 ), vec4( -1, -1, 0, 1 ), vec4( 1, 1, 0, 1 ), vec4( 1, -1, 0, 1 ) );


void main() 
{
    outUV = (gridPlane[gl_VertexIndex].xy + vec2(1.0f, 1.0f)) / 2.0f;
    gl_Position = gridPlane[gl_VertexIndex];
}
