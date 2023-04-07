#version 450

layout( push_constant ) uniform SceneUBO
{
    mat4 Model;
    mat4 View;
    mat4 Projection;
    vec4 Color;
}
Scene;

layout( location = 0 ) in vec3 inPosition;
layout( location = 1 ) in vec3 inNormal;

layout( location = 0 ) out vec3 outPosition;
layout( location = 1 ) out vec3 outNormal;

void main()
{
    gl_Position = Scene.Projection * Scene.View * Scene.Model * vec4( inPosition, 1.0 );
    outPosition = gl_Position.xyz;
    outNormal   = transpose( inverse( mat3( Scene.Model ) ) ) * inNormal;
}